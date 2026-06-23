from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_debye_wolf import gauss_theta_grid, relative_l2  # noqa: E402
from benchmark_high_na_gpu_dense_baseline import TorchDenseDirectVectorialDebyeWolf  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    make_pupil_batch,
    package_version,
    resolve_device,
    timed_torch,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_pupil_jones,
    workloads,
)


@dataclass
class CartesianInterpolationMap:
    torch: Any
    device: Any
    nrho: int
    npsi: int
    nz: int
    target_count: int
    r0: Any
    r1: Any
    p0: Any
    p1: Any
    z0: Any
    z1: Any
    wr0: Any
    wr1: Any
    wp0: Any
    wp1: Any
    wz0: Any
    wz1: Any

    @property
    def map_mib(self) -> float:
        tensors = (
            self.r0,
            self.r1,
            self.p0,
            self.p1,
            self.z0,
            self.z1,
            self.wr0,
            self.wr1,
            self.wp0,
            self.wp1,
            self.wz0,
            self.wz1,
        )
        return float(sum(int(t.nelement() * t.element_size()) for t in tensors) / (1024.0 * 1024.0))

    def forward(self, field: Any) -> Any:
        if field.ndim != 5:
            raise ValueError("field must have shape (batch, 3, nrho, npsi, nz)")

        def gather(r_idx: Any, p_idx: Any, z_idx: Any) -> Any:
            return field[:, :, r_idx, p_idx, z_idx]

        w000 = self.wr0 * self.wp0 * self.wz0
        w001 = self.wr0 * self.wp0 * self.wz1
        w010 = self.wr0 * self.wp1 * self.wz0
        w011 = self.wr0 * self.wp1 * self.wz1
        w100 = self.wr1 * self.wp0 * self.wz0
        w101 = self.wr1 * self.wp0 * self.wz1
        w110 = self.wr1 * self.wp1 * self.wz0
        w111 = self.wr1 * self.wp1 * self.wz1
        return (
            gather(self.r0, self.p0, self.z0) * w000[None, None, :]
            + gather(self.r0, self.p0, self.z1) * w001[None, None, :]
            + gather(self.r0, self.p1, self.z0) * w010[None, None, :]
            + gather(self.r0, self.p1, self.z1) * w011[None, None, :]
            + gather(self.r1, self.p0, self.z0) * w100[None, None, :]
            + gather(self.r1, self.p0, self.z1) * w101[None, None, :]
            + gather(self.r1, self.p1, self.z0) * w110[None, None, :]
            + gather(self.r1, self.p1, self.z1) * w111[None, None, :]
        )


def cartesian_axes(nx: int, ny: int, nz: int, half_width: float, z_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-half_width, half_width, nx, dtype=float)
    y_axis = np.linspace(-half_width, half_width, ny, dtype=float)
    if nz == 1:
        z_axis = np.array([0.0], dtype=float)
    else:
        z_axis = np.linspace(-z_max, z_max, nz, dtype=float)
    return x_axis, y_axis, z_axis


def cartesian_targets(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


def build_interpolation_map(
    *,
    torch: Any,
    device: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    dtype: str,
) -> CartesianInterpolationMap:
    real_dtype = torch.float32 if dtype == "complex64" else torch.float64
    rho = np.sqrt(x * x + y * y)
    psi = np.mod(np.arctan2(y, x), 2.0 * np.pi)

    rho_step = float(rho_axis[1] - rho_axis[0])
    rho_pos = np.clip(rho / rho_step, 0.0, rho_axis.size - 1.0)
    r0_np = np.floor(rho_pos).astype(np.int64)
    r0_np = np.clip(r0_np, 0, rho_axis.size - 2)
    r1_np = r0_np + 1
    ar_np = rho_pos - r0_np

    psi_step = float(2.0 * np.pi / psi_axis.size)
    psi_pos = psi / psi_step
    p0_np = np.floor(psi_pos).astype(np.int64) % psi_axis.size
    p1_np = (p0_np + 1) % psi_axis.size
    ap_np = psi_pos - np.floor(psi_pos)

    if z_axis.size == 1:
        z0_np = np.zeros_like(r0_np)
        z1_np = np.zeros_like(r0_np)
        az_np = np.zeros_like(rho_pos)
    else:
        z_step = float(z_axis[1] - z_axis[0])
        z_pos = np.clip((z - float(z_axis[0])) / z_step, 0.0, z_axis.size - 1.0)
        z0_np = np.floor(z_pos).astype(np.int64)
        z0_np = np.clip(z0_np, 0, z_axis.size - 2)
        z1_np = z0_np + 1
        az_np = z_pos - z0_np

    def idx(values: np.ndarray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.long, device=device)

    def weight(values: np.ndarray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=real_dtype, device=device)

    return CartesianInterpolationMap(
        torch=torch,
        device=device,
        nrho=int(rho_axis.size),
        npsi=int(psi_axis.size),
        nz=int(z_axis.size),
        target_count=int(x.size),
        r0=idx(r0_np),
        r1=idx(r1_np),
        p0=idx(p0_np),
        p1=idx(p1_np),
        z0=idx(z0_np),
        z1=idx(z1_np),
        wr0=weight(1.0 - ar_np),
        wr1=weight(ar_np),
        wp0=weight(1.0 - ap_np),
        wp1=weight(ap_np),
        wz0=weight(1.0 - az_np),
        wz1=weight(az_np),
    )


def intensity_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_i = np.sum(np.abs(reference) ** 2, axis=1)
    cand_i = np.sum(np.abs(candidate) ** 2, axis=1)
    return relative_l2(cand_i, ref_i)


def scale_fit_intensity_l2(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = np.sum(np.abs(reference) ** 2, axis=1).ravel().astype(float)
    cand = np.sum(np.abs(candidate) ** 2, axis=1).ravel().astype(float)
    denom = float(np.dot(cand, cand))
    if denom == 0.0:
        return float("inf"), 0.0
    scale = float(np.dot(ref, cand) / denom)
    err = float(np.linalg.norm(scale * cand - ref) / max(np.linalg.norm(ref), 1e-300))
    return err, scale


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


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
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA Cartesian interpolation adapter benchmark",
        "",
        "This benchmark tests whether the native cylindrical High-NA GPU solver can be used for Cartesian PSF output by evaluating an oversampled cylindrical grid and interpolating it to Cartesian target points.",
        "",
        "The reference is the same local vectorial dense direct Debye-Wolf CUDA implementation on the exact Cartesian target grid. This isolates interpolation/coordinate-adapter error before making any Cartesian package replacement claim.",
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- workload_index: `{config['workload_index']}`",
        f"- Cartesian grid: `{config['cartesian_nx']} x {config['cartesian_ny']} x {config['cartesian_nz']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- basis/contract: `{config['basis_mode']}` / `{config['contract_mode']}`",
        "",
        "## Results",
        "",
        "| oversample | cyl grid | cyl targets | dense fwd ms | eval ms | interp ms | eval+interp ms | speedup vs dense | complex L2 | intensity L2 | scale-fit intensity L2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {os} | {grid} | {targets} | {dense} | {eval_s} | {interp} | {total} | {speed} | {cl2} | {il2} | {sil2} |".format(
                os=row["oversample"],
                grid=f"{row['nrho']}x{row['npsi']}x{row['nz']}",
                targets=row["cylindrical_targets"],
                dense=fmt(1e3 * row["dense_cartesian_forward_hot_s"]),
                eval_s=fmt(1e3 * row["cylindrical_eval_hot_s"]),
                interp=fmt(1e3 * row["interpolation_hot_s"]),
                total=fmt(1e3 * row["eval_plus_interpolation_hot_s"]),
                speed=fmt(row["speedup_dense_vs_eval_plus_interp"]),
                cl2=fmt(row["complex_field_l2_vs_dense"]),
                il2=fmt(row["intensity_l2_vs_dense"]),
                sil2=fmt(row["scale_fit_intensity_l2_vs_dense"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Native cylindrical claims should use the cylindrical solver directly, without this interpolation adapter.",
            "- Cartesian package-replacement claims require this adapter error to be small at a useful oversampling factor.",
            "- Setup and map-build costs are reported in CSV/JSON and should be amortized for repeated masks or coherent modes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = import_torch()
    if torch is None:
        return [{"status": "skipped", "skip_reason": "torch is not installed"}]
    device = resolve_device(torch, args.device)
    workload = workloads(args.workload_set)[args.workload_index]

    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    x_axis, y_axis, cart_z_axis = cartesian_axes(
        args.cartesian_nx,
        args.cartesian_ny,
        args.cartesian_nz,
        args.cartesian_half_width,
        args.cartesian_z_max,
    )
    x, y, z = cartesian_targets(x_axis, y_axis, cart_z_axis)
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
    dense_forward, dense_forward_s, dense_times = timed_torch(
        torch,
        device,
        lambda: dense.forward(pupil_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    dense_np = to_numpy(torch, device, dense_forward)

    rows: list[dict[str, Any]] = []
    for oversample in parse_int_list(args.oversamples):
        nrho = int(args.base_nrho * oversample)
        npsi = int(args.base_npsi * oversample)
        nz = int(args.cylindrical_nz if args.cylindrical_nz > 0 else args.cartesian_nz)
        rho_max = float(args.cartesian_half_width * math.sqrt(2.0) * args.rho_margin)
        z_max = float(args.cartesian_z_max)
        rho_axis = np.linspace(0.0, rho_max, nrho, dtype=float)
        psi_axis = np.linspace(0.0, 2.0 * np.pi, npsi, endpoint=False, dtype=float)
        if nz == 1:
            z_axis = np.array([0.0], dtype=float)
        else:
            z_axis = np.linspace(-z_max, z_max, nz, dtype=float)
        h_cutoff = int(
            min(
                workload.nphi // 2,
                max(
                    0,
                    math.ceil(workload.k * rho_max * math.sin(workload.theta_max))
                    + abs(int(workload.vortex_charge))
                    + int(args.h_margin),
                ),
            )
        )
        setup_start = time.perf_counter()
        plan = TorchSeparableHarmonicDebyeWolfPlan.build(
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
        mixing_t = plan.as_tensor(mixing)
        interp = build_interpolation_map(
            torch=torch,
            device=device,
            x=x,
            y=y,
            z=z,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            dtype=args.dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        setup_s = time.perf_counter() - setup_start

        cyl_flat, eval_s, eval_times = timed_torch(
            torch,
            device,
            lambda: plan.evaluate_vectorial_batch(pupil_t, mixing_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        cyl_field = cyl_flat.reshape(args.batch_size, 3, nrho, npsi, nz)
        interp_value, interp_s, interp_times = timed_torch(
            torch,
            device,
            lambda: interp.forward(cyl_field),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        total_value, total_s, total_times = timed_torch(
            torch,
            device,
            lambda: interp.forward(
                plan.evaluate_vectorial_batch(pupil_t, mixing_t).reshape(
                    args.batch_size,
                    3,
                    nrho,
                    npsi,
                    nz,
                )
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        interp_np = to_numpy(torch, device, total_value)
        scale_l2, scale = scale_fit_intensity_l2(dense_np, interp_np)
        rows.append(
            {
                "status": "ok",
                "workload": workload.name,
                "workload_set": args.workload_set,
                "workload_index": args.workload_index,
                "device": str(device),
                "device_name": device_name(torch, device),
                "torch_version": package_version("torch"),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "dtype": args.dtype,
                "batch_size": int(args.batch_size),
                "cartesian_nx": int(args.cartesian_nx),
                "cartesian_ny": int(args.cartesian_ny),
                "cartesian_nz": int(args.cartesian_nz),
                "cartesian_targets": int(x.size),
                "oversample": int(oversample),
                "nrho": int(nrho),
                "npsi": int(npsi),
                "nz": int(nz),
                "rho_max": float(rho_max),
                "z_max": float(z_max),
                "cylindrical_targets": int(nrho * npsi * nz),
                "h_cutoff": int(h_cutoff),
                "used_modes": int(plan.used_modes),
                "basis_mode": plan.basis_mode,
                "contract_mode": plan.contract_mode,
                "basis_mib": float(plan.basis_mib),
                "interpolation_map_mib": float(interp.map_mib),
                "setup_s": float(setup_s),
                "dense_cartesian_forward_hot_s": float(dense_forward_s),
                "cylindrical_eval_hot_s": float(eval_s),
                "interpolation_hot_s": float(interp_s),
                "eval_plus_interpolation_hot_s": float(total_s),
                "speedup_dense_vs_eval_plus_interp": float(dense_forward_s / total_s),
                "complex_field_l2_vs_dense": float(relative_l2(interp_np, dense_np)),
                "intensity_l2_vs_dense": float(intensity_l2(dense_np, interp_np)),
                "scale_fit_intensity_l2_vs_dense": float(scale_l2),
                "scale_fit_intensity_scale": float(scale),
                "dense_forward_times_s": " ".join(f"{value:.9g}" for value in dense_times),
                "cylindrical_eval_times_s": " ".join(f"{value:.9g}" for value in eval_times),
                "interpolation_times_s": " ".join(f"{value:.9g}" for value in interp_times),
                "eval_plus_interpolation_times_s": " ".join(f"{value:.9g}" for value in total_times),
                "gpu_peak_allocated_mib": None
                if device.type != "cuda"
                else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            }
        )
        del cyl_flat, cyl_field, interp_value, total_value

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cylindrical High-NA GPU evaluation plus interpolation to Cartesian PSF targets."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_cartesian_interpolation_adapter")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-phase-strength", type=float, default=0.25)
    parser.add_argument("--cartesian-nx", type=int, default=32)
    parser.add_argument("--cartesian-ny", type=int, default=32)
    parser.add_argument("--cartesian-nz", type=int, default=5)
    parser.add_argument("--cartesian-half-width", type=float, default=2.0)
    parser.add_argument("--cartesian-z-max", type=float, default=1.0)
    parser.add_argument("--base-nrho", type=int, default=32)
    parser.add_argument("--base-npsi", type=int, default=96)
    parser.add_argument("--cylindrical-nz", type=int, default=0)
    parser.add_argument("--oversamples", default="1,2")
    parser.add_argument("--rho-margin", type=float, default=1.001)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="fused")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="matmul")
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--chunk-targets", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
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
