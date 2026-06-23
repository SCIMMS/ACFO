from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
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
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import (  # noqa: E402
    complex_dot,
    focal_grid,
    relative_complex_error,
)
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


@dataclass
class CuFinufftDebyeWolfPlan:
    cp: Any
    cufinufft: Any
    complex_dtype: Any
    real_dtype: Any
    np_complex_dtype: Any
    np_real_dtype: Any
    dtype: str
    ntheta: int
    nphi: int
    targets: int
    n_trans: int
    coord_scale: float
    source_weight: np.ndarray
    source_x: Any
    source_y: Any
    source_z: Any
    target_s: Any
    target_t: Any
    target_u: Any
    forward_plan: Any
    adjoint_plan: Any
    forward_out: Any
    adjoint_out: Any

    @property
    def coordinate_mib(self) -> float:
        bytes_total = 0
        for array in (
            self.source_x,
            self.source_y,
            self.source_z,
            self.target_s,
            self.target_t,
            self.target_u,
        ):
            bytes_total += int(array.nbytes)
        bytes_total += int(self.source_weight.nbytes)
        return float(bytes_total / (1024.0 * 1024.0))

    @property
    def output_mib(self) -> float:
        bytes_total = int(self.forward_out.nbytes + self.adjoint_out.nbytes)
        return float(bytes_total / (1024.0 * 1024.0))

    @classmethod
    def build(
        cls,
        *,
        theta: np.ndarray,
        theta_weights: np.ndarray,
        phi: np.ndarray,
        rho: np.ndarray,
        psi: np.ndarray,
        z: np.ndarray,
        k: float,
        dtype: str,
        eps: float,
        n_trans: int,
    ) -> "CuFinufftDebyeWolfPlan":
        cp, cufinufft = import_cufinufft_modules()
        complex_dtype, real_dtype, np_complex_dtype, np_real_dtype = cupy_dtypes(cp, dtype)

        theta_2d = theta[:, None]
        phi_2d = phi[None, :]
        xi_x = k * np.sin(theta_2d) * np.cos(phi_2d)
        xi_y = k * np.sin(theta_2d) * np.sin(phi_2d)
        xi_z = np.broadcast_to(k * np.cos(theta_2d), xi_x.shape)
        max_abs_xi = float(max(np.max(np.abs(xi_x)), np.max(np.abs(xi_y)), np.max(np.abs(xi_z))))
        coord_scale = max(max_abs_xi / (0.95 * np.pi), 1.0)

        dphi = 2.0 * np.pi / float(phi.size)
        source_weight = np.broadcast_to(
            theta_weights[:, None] * dphi,
            (theta.size, phi.size),
        ).copy()

        source_x = cp.asarray(np.ascontiguousarray((xi_x / coord_scale).ravel().astype(np_real_dtype)))
        source_y = cp.asarray(np.ascontiguousarray((xi_y / coord_scale).ravel().astype(np_real_dtype)))
        source_z = cp.asarray(np.ascontiguousarray((xi_z / coord_scale).ravel().astype(np_real_dtype)))
        target_s = cp.asarray(np.ascontiguousarray((rho * np.cos(psi) * coord_scale).astype(np_real_dtype)))
        target_t = cp.asarray(np.ascontiguousarray((rho * np.sin(psi) * coord_scale).astype(np_real_dtype)))
        target_u = cp.asarray(np.ascontiguousarray((z * coord_scale).astype(np_real_dtype)))

        forward_plan = cufinufft.Plan(
            3,
            3,
            n_trans=int(n_trans),
            eps=eps,
            isign=1,
            dtype=dtype,
        )
        forward_plan.setpts(source_x, source_y, source_z, target_s, target_t, target_u)
        adjoint_plan = cufinufft.Plan(
            3,
            3,
            n_trans=int(n_trans),
            eps=eps,
            isign=-1,
            dtype=dtype,
        )
        adjoint_plan.setpts(target_s, target_t, target_u, source_x, source_y, source_z)
        forward_out = cp.empty((int(n_trans), int(rho.size)), dtype=complex_dtype)
        adjoint_out = cp.empty((int(n_trans), int(theta.size * phi.size)), dtype=complex_dtype)
        cp.cuda.Stream.null.synchronize()

        return cls(
            cp=cp,
            cufinufft=cufinufft,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            np_complex_dtype=np_complex_dtype,
            np_real_dtype=np_real_dtype,
            dtype=dtype,
            ntheta=int(theta.size),
            nphi=int(phi.size),
            targets=int(rho.size),
            n_trans=int(n_trans),
            coord_scale=float(coord_scale),
            source_weight=np.ascontiguousarray(source_weight.astype(np_real_dtype)),
            source_x=source_x,
            source_y=source_y,
            source_z=source_z,
            target_s=target_s,
            target_t=target_t,
            target_u=target_u,
            forward_plan=forward_plan,
            adjoint_plan=adjoint_plan,
            forward_out=forward_out,
            adjoint_out=adjoint_out,
        )

    def strengths_gpu(self, pupil_batch: np.ndarray, mixing: np.ndarray) -> Any:
        effective = np.einsum("cjtp,bjtp->bctp", mixing, pupil_batch, optimize=True)
        strengths = effective.reshape(-1, self.ntheta, self.nphi) * self.source_weight[None, :, :]
        return self.cp.asarray(
            np.ascontiguousarray(
                strengths.reshape(self.n_trans, self.ntheta * self.nphi).astype(
                    self.np_complex_dtype,
                    copy=False,
                )
            )
        )

    def residual_gpu(self, residual_batch: np.ndarray) -> Any:
        residuals = np.asarray(residual_batch)
        if residuals.shape != (self.n_trans // 3, 3, self.targets):
            raise ValueError("residual batch shape does not match cuFINUFFT plan")
        return self.cp.asarray(
            np.ascontiguousarray(
                residuals.reshape(self.n_trans, self.targets).astype(
                    self.np_complex_dtype,
                    copy=False,
                )
            )
        )

    def forward(self, strengths_gpu: Any) -> Any:
        return self.forward_plan.execute(strengths_gpu, out=self.forward_out)

    def adjoint_effective(self, residual_gpu: Any) -> Any:
        return self.adjoint_plan.execute(residual_gpu, out=self.adjoint_out)

    def forward_numpy(self, value_gpu: Any) -> np.ndarray:
        self.cp.cuda.Stream.null.synchronize()
        return self.cp.asnumpy(value_gpu).reshape(self.n_trans // 3, 3, self.targets)

    def adjoint_numpy(self, value_gpu: Any, mixing: np.ndarray) -> np.ndarray:
        self.cp.cuda.Stream.null.synchronize()
        effective = self.cp.asnumpy(value_gpu).reshape(
            self.n_trans // 3,
            3,
            self.ntheta,
            self.nphi,
        )
        effective *= self.source_weight[None, None, :, :]
        return np.einsum("cjtp,bctp->bjtp", np.conjugate(mixing), effective, optimize=True)


def add_gpu_dll_directories() -> None:
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cupy_cache"))
    for path in (
        ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
        ROOT / ".venv" / "Lib" / "site-packages" / "cufinufft",
        ROOT / ".venv" / "Lib" / "site-packages" / "cufinufft.libs",
    ):
        if path.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


def import_cufinufft_modules() -> tuple[Any, Any]:
    add_gpu_dll_directories()
    try:
        import cupy as cp
        import cufinufft
    except ImportError as exc:
        raise RuntimeError("cupy/cufinufft is not installed or cannot load CUDA DLLs") from exc
    return cp, cufinufft


def cupy_dtypes(cp: Any, dtype: str) -> tuple[Any, Any, Any, Any]:
    if dtype == "complex64":
        return cp.complex64, cp.float32, np.complex64, np.float32
    if dtype == "complex128":
        return cp.complex128, cp.float64, np.complex128, np.float64
    raise ValueError("dtype must be complex64 or complex128")


def cufinufft_version() -> str:
    try:
        return importlib.metadata.version("cufinufft")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def cupy_version() -> str:
    try:
        add_gpu_dll_directories()
        import cupy as cp

        return str(cp.__version__)
    except Exception:
        pass
    try:
        return importlib.metadata.version("cupy")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def timed_cupy(cp: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
        cp.cuda.Stream.null.synchronize()
    times: list[float] = []
    for _ in range(max(1, repeats)):
        gc.collect()
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        value = func()
        cp.cuda.Stream.null.synchronize()
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
        writer.writerows(rows)


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA GPU cuFINUFFT baseline",
        "",
        "This benchmark compares the structured Torch/CUDA vectorial High-NA separable path with a matched cuFINUFFT type-3 baseline on the same cylindrical target grid.",
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- workload_index: `{config['workload_index']}`",
        f"- dtype: `{config['dtype']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- cuFINUFFT eps: `{config['cufinufft_eps']}`",
        f"- torch basis/contract: `{config['basis_mode']}` / `{config['contract_mode']}`",
        "",
        "## Results",
        "",
        "| workload | batch | targets | transforms | torch pair ms | cuFINUFFT pair ms | pair speedup | torch fwd ms | cuFINUFFT fwd ms | torch adj ms | cuFINUFFT adj ms | fwd L2 | adj L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('status')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            lines.append("")
            lines.append(f"Skip reason: `{row.get('skip_reason')}`")
            continue
        lines.append(
            "| {workload} | {batch} | {targets} | {trans} | {torch_pair} | {cu_pair} | {speed} | {torch_fwd} | {cu_fwd} | {torch_adj} | {cu_adj} | {fwd_l2} | {adj_l2} |".format(
                workload=row["workload"],
                batch=row["batch_size"],
                targets=row["targets_per_mask"],
                trans=row["n_transforms"],
                torch_pair=fmt(1e3 * row["torch_forward_plus_adjoint_hot_s"]),
                cu_pair=fmt(1e3 * row["cufinufft_forward_plus_adjoint_hot_s"]),
                speed=fmt(row["cufinufft_over_torch_pair_speedup"]),
                torch_fwd=fmt(1e3 * row["torch_forward_hot_s"]),
                cu_fwd=fmt(1e3 * row["cufinufft_forward_hot_s"]),
                torch_adj=fmt(1e3 * row["torch_adjoint_hot_s"]),
                cu_adj=fmt(1e3 * row["cufinufft_adjoint_hot_s"]),
                fwd_l2=fmt(row["torch_forward_l2_vs_cufinufft"]),
                adj_l2=fmt(row["torch_adjoint_l2_vs_cufinufft"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- cuFINUFFT is a generic type-3 NUFFT baseline, not a domain-specific FFT-Debye solver.",
            "- Hot timings exclude host-to-device input transfer for both methods.",
            "- The cuFINUFFT baseline batches the `batch_size * 3` scalar vectorial component transforms into one plan.",
            "- Current CuPy installation can run cuFINUFFT, but generic CuPy elementwise/linalg kernels require CUDA headers; this script avoids those kernels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def skipped_row(args: argparse.Namespace, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "status": "skipped",
            "skip_reason": reason,
            "workload_set": args.workload_set,
            "workload_index": args.workload_index,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "torch_version": package_version("torch"),
            "cufinufft_version": cufinufft_version(),
            "cupy_version": cupy_version(),
        }
    ]


def run_case(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = import_torch()
    if torch is None:
        return skipped_row(args, "torch is not installed")
    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        return skipped_row(args, str(exc))
    if device.type != "cuda":
        return skipped_row(args, "CUDA torch device is required for this comparison")

    try:
        cp, _ = import_cufinufft_modules()
    except RuntimeError as exc:
        return skipped_row(args, str(exc))

    workload = workloads(args.workload_set)[args.workload_index]
    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, rho, psi, z = focal_grid(
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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(torch, device)
    torch_setup_start = time.perf_counter()
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
        device=device,
        dtype=args.dtype,
        basis_mode=args.basis_mode,
        contract_mode=args.contract_mode,
    )
    pupil_t = torch_plan.as_tensor(pupil_batch)
    residual_t = torch_plan.as_tensor(residual_batch)
    mixing_t = torch_plan.as_tensor(mixing)
    synchronize(torch, device)
    torch_setup_s = time.perf_counter() - torch_setup_start

    torch_forward, torch_forward_s, torch_forward_times = timed_torch(
        torch,
        device,
        lambda: torch_plan.evaluate_vectorial_batch(pupil_t, mixing_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    torch_adjoint, torch_adjoint_s, torch_adjoint_times = timed_torch(
        torch,
        device,
        lambda: torch_plan.adjoint_vectorial_batch(residual_t, mixing_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )

    n_trans = int(args.batch_size * 3)
    cp.cuda.Stream.null.synchronize()
    cufinufft_setup_start = time.perf_counter()
    cu_plan = CuFinufftDebyeWolfPlan.build(
        theta=theta,
        theta_weights=theta_weights,
        phi=phi,
        rho=rho,
        psi=psi,
        z=z,
        k=workload.k,
        dtype=args.dtype,
        eps=args.cufinufft_eps,
        n_trans=n_trans,
    )
    strengths_gpu = cu_plan.strengths_gpu(pupil_batch, mixing)
    residual_gpu = cu_plan.residual_gpu(residual_batch)
    cp.cuda.Stream.null.synchronize()
    cufinufft_setup_s = time.perf_counter() - cufinufft_setup_start

    cu_forward, cu_forward_s, cu_forward_times = timed_cupy(
        cp,
        lambda: cu_plan.forward(strengths_gpu),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    cu_adjoint_effective, cu_adjoint_s, cu_adjoint_times = timed_cupy(
        cp,
        lambda: cu_plan.adjoint_effective(residual_gpu),
        repeats=args.repeats,
        warmups=args.warmups,
    )

    torch_forward_np = to_numpy(torch, device, torch_forward)
    torch_adjoint_np = to_numpy(torch, device, torch_adjoint)
    cu_forward_np = cu_plan.forward_numpy(cu_forward)
    cu_adjoint_np = cu_plan.adjoint_numpy(cu_adjoint_effective, mixing)

    torch_dot_left = complex_dot(torch_forward_np, residual_batch)
    torch_dot_right = complex_dot(pupil_batch, torch_adjoint_np)
    cu_dot_left = complex_dot(cu_forward_np, residual_batch)
    cu_dot_right = complex_dot(pupil_batch, cu_adjoint_np)

    return [
        {
            "status": "ok",
            "workload": workload.name,
            "workload_set": args.workload_set,
            "workload_index": args.workload_index,
            "device": str(device),
            "device_name": device_name(torch, device),
            "torch_version": package_version("torch"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cupy_version": cupy_version(),
            "cufinufft_version": cufinufft_version(),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "n_transforms": n_trans,
            "ntheta": workload.ntheta,
            "nphi": workload.nphi,
            "nrho": workload.nrho,
            "npsi": workload.npsi,
            "nz": workload.nz,
            "targets_per_mask": int(rho.size),
            "source_samples": int(workload.ntheta * workload.nphi),
            "h_cutoff": int(h_cutoff),
            "used_modes": int(torch_plan.used_modes),
            "basis_mode": torch_plan.basis_mode,
            "contract_mode": torch_plan.contract_mode,
            "torch_basis_mib": float(torch_plan.basis_mib),
            "cufinufft_coordinate_mib": float(cu_plan.coordinate_mib),
            "cufinufft_output_mib": float(cu_plan.output_mib),
            "cufinufft_coord_scale": float(cu_plan.coord_scale),
            "cufinufft_eps": float(args.cufinufft_eps),
            "torch_setup_s": float(torch_setup_s),
            "cufinufft_setup_s": float(cufinufft_setup_s),
            "torch_forward_hot_s": float(torch_forward_s),
            "torch_adjoint_hot_s": float(torch_adjoint_s),
            "torch_forward_plus_adjoint_hot_s": float(torch_forward_s + torch_adjoint_s),
            "cufinufft_forward_hot_s": float(cu_forward_s),
            "cufinufft_adjoint_hot_s": float(cu_adjoint_s),
            "cufinufft_forward_plus_adjoint_hot_s": float(cu_forward_s + cu_adjoint_s),
            "cufinufft_over_torch_forward_speedup": float(cu_forward_s / torch_forward_s),
            "cufinufft_over_torch_adjoint_speedup": float(cu_adjoint_s / torch_adjoint_s),
            "cufinufft_over_torch_pair_speedup": float(
                (cu_forward_s + cu_adjoint_s) / (torch_forward_s + torch_adjoint_s)
            ),
            "cufinufft_over_torch_pair_speedup_with_setup": float(
                (cufinufft_setup_s + cu_forward_s + cu_adjoint_s)
                / (torch_setup_s + torch_forward_s + torch_adjoint_s)
            ),
            "torch_forward_l2_vs_cufinufft": float(relative_l2(torch_forward_np, cu_forward_np)),
            "torch_adjoint_l2_vs_cufinufft": float(relative_l2(torch_adjoint_np, cu_adjoint_np)),
            "torch_dot_relative_error": relative_complex_error(torch_dot_left, torch_dot_right),
            "cufinufft_dot_relative_error": relative_complex_error(cu_dot_left, cu_dot_right),
            "cross_dot_left_relative_error": relative_complex_error(torch_dot_left, cu_dot_left),
            "cross_dot_right_relative_error": relative_complex_error(torch_dot_right, cu_dot_right),
            "gpu_peak_allocated_mib_torch": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            "torch_forward_times_s": " ".join(f"{value:.9g}" for value in torch_forward_times),
            "torch_adjoint_times_s": " ".join(f"{value:.9g}" for value in torch_adjoint_times),
            "cufinufft_forward_times_s": " ".join(f"{value:.9g}" for value in cu_forward_times),
            "cufinufft_adjoint_times_s": " ".join(f"{value:.9g}" for value in cu_adjoint_times),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
            "apodization": args.apodization,
            "batch_phase_strength": float(args.batch_phase_strength),
            "residual_case": args.residual_case,
            "residual_order": int(args.residual_order),
            "seed": int(args.seed),
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Torch separable High-NA GPU propagation with a cuFINUFFT type-3 baseline."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_gpu_cufinufft_baseline")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-phase-strength", type=float, default=0.25)
    parser.add_argument("--residual-case", choices=["random", "low_order", "annular_roi"], default="low_order")
    parser.add_argument("--residual-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="fused")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="matmul")
    parser.add_argument("--cufinufft-eps", type=float, default=1e-6)
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
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
