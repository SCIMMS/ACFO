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
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

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
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    cufinufft_adjoint_block,
    cufinufft_forward_block,
    cupy_dtypes,
    import_cufinufft_modules,
    make_block,
    synchronize_cupy,
)
from benchmark_odt_iterative_reconstruction import build_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchConeAxisOdtPlan,
    rel_l2,
    torch_dtypes,
    to_numpy,
)
from waxs_cake.odt_measured_contract import (  # noqa: E402
    load_odt_measured_contract,
    validate_odt_measured_contract,
)


@dataclass(frozen=True)
class AIDTResidual:
    residual: np.ndarray
    frame_order: np.ndarray
    detector_na: float
    illumination_na: float
    fft_norm: str
    cap_radial: int
    cap_phi: int

    @property
    def q_count(self) -> int:
        return int(self.residual.size)


def centered_fft2(image: np.ndarray, *, norm: str) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm=None if norm == "none" else norm))


def uniform_ring_frame_order(source_na_xy: np.ndarray) -> np.ndarray:
    source_phi = np.mod(np.arctan2(source_na_xy[:, 1], source_na_xy[:, 0]), 2.0 * np.pi)
    target_phi = np.linspace(0.0, 2.0 * np.pi, source_phi.size, endpoint=False)
    used: set[int] = set()
    order: list[int] = []
    for phi in target_phi:
        distances = np.abs(np.angle(np.exp(1j * (source_phi - phi))))
        for index in np.argsort(distances):
            idx = int(index)
            if idx not in used:
                used.add(idx)
                order.append(idx)
                break
    if len(order) != source_phi.size:
        raise ValueError("could not build a unique frame order for the annular source ring")
    return np.asarray(order, dtype=np.int64)


def interpolate_fft_on_polar_detector(
    fft_image: np.ndarray,
    *,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    wavelength_um: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    interpolation_order: int,
) -> np.ndarray:
    radial_na = (np.arange(cap_radial, dtype=np.float64) + 0.5) * float(detector_na) / float(cap_radial)
    phi = np.linspace(0.0, 2.0 * np.pi, cap_phi, endpoint=False, dtype=np.float64)
    rr, pp = np.meshgrid(radial_na, phi, indexing="ij")
    fx = rr * np.cos(pp) / float(wavelength_um)
    fy = rr * np.sin(pp) / float(wavelength_um)
    if not (np.all(np.diff(frequency_x) > 0.0) and np.all(np.diff(frequency_y) > 0.0)):
        raise ValueError("frequency axes must be strictly increasing")
    x_coord = (fx - frequency_x[0]) / (frequency_x[-1] - frequency_x[0]) * (frequency_x.size - 1)
    y_coord = (fy - frequency_y[0]) / (frequency_y[-1] - frequency_y[0]) * (frequency_y.size - 1)
    coords = np.vstack([y_coord.ravel(), x_coord.ravel()])
    real = map_coordinates(
        np.real(fft_image),
        coords,
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
    )
    imag = map_coordinates(
        np.imag(fft_image),
        coords,
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
    )
    return (real + 1j * imag).reshape(cap_radial, cap_phi)


def measured_polar_residual_from_contract(
    contract_path: Path,
    *,
    cap_radial: int,
    cap_phi: int,
    detector_na: float | None,
    fft_norm: str,
    interpolation_order: int,
    normalize: bool,
    max_frames: int | None,
) -> tuple[AIDTResidual, dict[str, Any]]:
    measured = load_odt_measured_contract(contract_path)
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    if measured.q_layout != "annular_cartesian_stack":
        raise ValueError(f"expected annular_cartesian_stack contract, got {measured.q_layout!r}")
    data = np.asarray(measured["data"], dtype=np.float32)
    source_na_xy = np.asarray(measured["source_na_xy"], dtype=np.float64)
    frequency_x = np.asarray(measured["frequency_x"], dtype=np.float64)
    frequency_y = np.asarray(measured["frequency_y"], dtype=np.float64)
    wavelength_um = float(measured["wavelength"])
    objective_na = float(measured["objective_na"])
    detector_na_value = float(objective_na if detector_na is None else detector_na)
    frame_order = uniform_ring_frame_order(source_na_xy)
    if max_frames is not None:
        frame_order = frame_order[: int(max_frames)]
    blocks = []
    for frame_index in frame_order:
        fft_image = centered_fft2(data[int(frame_index)], norm=fft_norm)
        blocks.append(
            interpolate_fft_on_polar_detector(
                fft_image,
                frequency_x=frequency_x,
                frequency_y=frequency_y,
                wavelength_um=wavelength_um,
                detector_na=detector_na_value,
                cap_radial=cap_radial,
                cap_phi=cap_phi,
                interpolation_order=interpolation_order,
            )
        )
    residual = np.ascontiguousarray(np.stack(blocks, axis=0).reshape(-1).astype(np.complex64))
    raw_norm = float(np.linalg.norm(residual.astype(np.complex128)))
    if normalize and raw_norm > 0.0:
        residual = np.ascontiguousarray((residual / (raw_norm / math.sqrt(residual.size))).astype(np.complex64))
    source_radii = np.linalg.norm(source_na_xy[frame_order], axis=1)
    metadata = {
        "contract_path": str(contract_path),
        "validation": report.to_dict(),
        "raw_data_shape": tuple(int(v) for v in data.shape),
        "frame_order": [int(v) for v in frame_order],
        "wavelength_um": wavelength_um,
        "objective_na": objective_na,
        "detector_na": detector_na_value,
        "illumination_na_mean": float(np.mean(source_radii)),
        "illumination_na_std": float(np.std(source_radii)),
        "residual_raw_norm": raw_norm,
        "residual_norm": float(np.linalg.norm(residual.astype(np.complex128))),
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "residual_abs_max": float(np.max(np.abs(residual))),
        "fft_norm": fft_norm,
        "interpolation_order": int(interpolation_order),
        "residual_normalized": bool(normalize),
    }
    return (
        AIDTResidual(
            residual=residual,
            frame_order=frame_order,
            detector_na=detector_na_value,
            illumination_na=float(np.mean(source_radii)),
            fft_norm=fft_norm,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
        ),
        metadata,
    )


def timed_torch(torch: Any, device: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
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
        raise RuntimeError("timed torch function did not run")
    return value, float(median(times)), times


def timed_cupy(cp: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
        synchronize_cupy(cp)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        synchronize_cupy(cp)
        start = time.perf_counter()
        value = func()
        synchronize_cupy(cp)
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed CuPy function did not run")
    return value, float(median(times)), times


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Public aIDT measured-data adapter benchmark",
        "",
        "This benchmark converts the public aIDT Diatom I intensity stack into a Fourier-domain polar residual and applies the same measured residual to the prepared cone-axis GPU adjoint and a cuFINUFFT type-3 GPU adjoint baseline.",
        "",
        "## Data Adapter",
        "",
        f"- contract: `{summary['contract_path']}`",
        f"- raw data shape: `{summary['raw_data_shape']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- illumination frames used: `{summary['n_illum']}`",
        f"- measured residual samples: `{summary['q_samples']}`",
        f"- detector NA: `{summary['detector_na']}`",
        f"- illumination NA mean/std: `{summary['illumination_na_mean']:.6g}` / `{summary['illumination_na_std']:.6g}`",
        f"- FFT normalization: `{summary['fft_norm']}`",
        f"- interpolation order: `{summary['interpolation_order']}`",
        "",
        "## Operator Timing",
        "",
        "| path | median ms | speedup vs cuFINUFFT |",
        "| --- | ---: | ---: |",
        f"| prepared measured adjoint | {1000.0 * summary['ours_measured_adjoint_s']:.3f} | {summary['cufinufft_measured_adjoint_s'] / summary['ours_measured_adjoint_s']:.3f}x |",
        f"| cuFINUFFT measured adjoint | {1000.0 * summary['cufinufft_measured_adjoint_s']:.3f} | 1.000x |",
        f"| prepared forward | {1000.0 * summary['ours_forward_s']:.3f} | {summary['cufinufft_forward_s'] / summary['ours_forward_s']:.3f}x |",
        f"| cuFINUFFT forward | {1000.0 * summary['cufinufft_forward_s']:.3f} | 1.000x |",
        f"| prepared measured adjoint+forward | {1000.0 * summary['ours_measured_pair_s']:.3f} | {summary['cufinufft_measured_pair_s'] / summary['ours_measured_pair_s']:.3f}x |",
        f"| cuFINUFFT measured adjoint+forward | {1000.0 * summary['cufinufft_measured_pair_s']:.3f} | 1.000x |",
        "",
        "## Accuracy",
        "",
        f"- forward rel-L2, cuFINUFFT vs prepared: `{summary['cufinufft_forward_rel_l2_vs_ours']:.6g}`",
        f"- measured adjoint rel-L2, cuFINUFFT vs prepared: `{summary['cufinufft_measured_adjoint_rel_l2_vs_ours']:.6g}`",
        f"- measured pair rel-L2, cuFINUFFT vs prepared: `{summary['cufinufft_measured_pair_rel_l2_vs_ours']:.6g}`",
        "",
        "## Interpretation",
        "",
        "- This is the first real public-data adapter benchmark, not a full physical aIDT reconstruction.",
        "- The measured data enter through the residual used by the adjoint. The forward parity check still uses a synthetic object on the same measured-data geometry.",
        "- The next step is to replace this adjoint/pair smoke benchmark with a measured-data update loop and then add calibration/mask handling.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_operator_args(args: argparse.Namespace, residual: AIDTResidual) -> argparse.Namespace:
    return argparse.Namespace(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
        k=args.k,
        detector_na=residual.detector_na,
        cap_radial=residual.cap_radial,
        cap_phi=residual.cap_phi,
        n_illum=int(residual.frame_order.size),
        illumination_na=residual.illumination_na,
        h_cutoff=args.h_cutoff,
        h_margin=args.h_margin,
        l_margin=args.l_margin,
        cone_l_prune_threshold=args.cone_l_prune_threshold,
        native_prepared_plan_mode=args.native_prepared_plan_mode,
        native_prepared_gather_threshold=args.native_prepared_gather_threshold,
        cpp_threads=args.cpp_threads,
        forward_execute_mode="prepared",
        forward_kernel_mode=args.forward_kernel_mode,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    measured_residual, adapter_meta = measured_polar_residual_from_contract(
        args.contract,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        detector_na=args.detector_na,
        fft_norm=args.fft_norm,
        interpolation_order=args.interpolation_order,
        normalize=not args.no_residual_normalize,
        max_frames=args.max_frames,
    )
    op_args = build_operator_args(args, measured_residual)
    ctx = build_context(op_args)

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cp, cufinufft = import_cufinufft_modules()

    start = time.perf_counter()
    torch_plan = TorchConeAxisOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    ours_setup_s = time.perf_counter() - start

    complex_dtype, _, np_complex, np_real = torch_dtypes(torch, args.dtype)
    cu_complex_dtype, _, _, _ = cupy_dtypes(cp, args.dtype)
    simple_ctx = SimpleNamespace(obj=ctx.obj, flat_q=ctx.flat_q)
    start = time.perf_counter()
    cu_block = make_block(
        cp,
        cufinufft,
        simple_ctx,
        dtype=args.dtype,
        complex_dtype=cu_complex_dtype,
        np_real=np_real,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cu_setup_s = time.perf_counter() - start
    cu_op = SimpleNamespace(cp=cp, cufinufft=cufinufft)

    true_coeff_np = np.ascontiguousarray(ctx.obj.coeff.astype(np_complex, copy=False) * float(args.object_scale))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=complex_dtype, device=device)
    coeff_gpu = cp.asarray(true_coeff_np.ravel())
    residual_np = measured_residual.residual.astype(np_complex, copy=False)
    residual_t = torch.as_tensor(residual_np, dtype=complex_dtype, device=device)
    residual_gpu = cp.asarray(residual_np)

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        ours_forward_t, ours_forward_s, ours_forward_times = timed_torch(
            torch,
            device,
            lambda: torch_plan.forward(true_coeff_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        ours_adjoint_t, ours_adjoint_s, ours_adjoint_times = timed_torch(
            torch,
            device,
            lambda: torch_plan.adjoint(residual_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        ours_pair_t, ours_pair_s, ours_pair_times = timed_torch(
            torch,
            device,
            lambda: torch_plan.forward(torch_plan.adjoint(residual_t)),
            repeats=args.repeats,
            warmups=args.warmups,
        )

    cu_forward_gpu, cu_forward_s, cu_forward_times = timed_cupy(
        cp,
        lambda: cufinufft_forward_block(cu_op, cu_block, coeff_gpu, eps=args.cufinufft_eps),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    cu_forward_np = np.asarray(cu_forward_gpu.get(), dtype=np.complex128)
    cu_adjoint_gpu, cu_adjoint_s, cu_adjoint_times = timed_cupy(
        cp,
        lambda: cufinufft_adjoint_block(cu_op, cu_block, residual_gpu, eps=args.cufinufft_eps),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    cu_adjoint_np = np.asarray(cu_adjoint_gpu.get(), dtype=np.complex128).reshape(true_coeff_np.shape)
    cu_pair_gpu, cu_pair_s, cu_pair_times = timed_cupy(
        cp,
        lambda: cufinufft_forward_block(
            cu_op,
            cu_block,
            cufinufft_adjoint_block(cu_op, cu_block, residual_gpu, eps=args.cufinufft_eps),
            eps=args.cufinufft_eps,
        ),
        repeats=args.repeats,
        warmups=args.warmups,
    )

    ours_forward_np = to_numpy(torch, device, ours_forward_t).astype(np.complex128, copy=False)
    ours_adjoint_np = to_numpy(torch, device, ours_adjoint_t).astype(np.complex128, copy=False)
    ours_pair_np = to_numpy(torch, device, ours_pair_t).astype(np.complex128, copy=False)
    cu_pair_np = np.asarray(cu_pair_gpu.get(), dtype=np.complex128)

    summary = {
        **adapter_meta,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cufinufft, "__version__", None),
        "dtype": args.dtype,
        "cufinufft_eps": float(args.cufinufft_eps),
        "cufinufft_plan_mode": args.cufinufft_plan_mode,
        "k": float(args.k),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "n_illum": int(measured_residual.frame_order.size),
        "q_samples": int(measured_residual.q_count),
        "object_bins": int(true_coeff_np.size),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "n_beta": int(args.n_beta),
        "h_modes": int(ctx.plan.used_modes),
        "l_cutoff": int(ctx.l_cutoff),
        "axis_h_cutoff": int(ctx.axis_h_cutoff),
        "ours_setup_s": float(ours_setup_s),
        "cufinufft_setup_s": float(cu_setup_s),
        "ours_measured_adjoint_s": float(ours_adjoint_s),
        "cufinufft_measured_adjoint_s": float(cu_adjoint_s),
        "ours_forward_s": float(ours_forward_s),
        "cufinufft_forward_s": float(cu_forward_s),
        "ours_measured_pair_s": float(ours_pair_s),
        "cufinufft_measured_pair_s": float(cu_pair_s),
        "ours_speedup_vs_cufinufft_measured_adjoint": float(cu_adjoint_s / ours_adjoint_s),
        "ours_speedup_vs_cufinufft_forward": float(cu_forward_s / ours_forward_s),
        "ours_speedup_vs_cufinufft_measured_pair": float(cu_pair_s / ours_pair_s),
        "cufinufft_forward_rel_l2_vs_ours": rel_l2(cu_forward_np, ours_forward_np),
        "cufinufft_measured_adjoint_rel_l2_vs_ours": rel_l2(cu_adjoint_np, ours_adjoint_np),
        "cufinufft_measured_pair_rel_l2_vs_ours": rel_l2(cu_pair_np, ours_pair_np),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "cupy_pool_total_mib": float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
        "history_csv": str(args.csv),
        "summary_md": str(args.summary_md),
    }
    rows = [
        {"path": "ours_forward", "median_s": ours_forward_s, "times_s": " ".join(f"{v:.9g}" for v in ours_forward_times)},
        {"path": "ours_measured_adjoint", "median_s": ours_adjoint_s, "times_s": " ".join(f"{v:.9g}" for v in ours_adjoint_times)},
        {"path": "ours_measured_pair", "median_s": ours_pair_s, "times_s": " ".join(f"{v:.9g}" for v in ours_pair_times)},
        {"path": "cufinufft_forward", "median_s": cu_forward_s, "times_s": " ".join(f"{v:.9g}" for v in cu_forward_times)},
        {"path": "cufinufft_measured_adjoint", "median_s": cu_adjoint_s, "times_s": " ".join(f"{v:.9g}" for v in cu_adjoint_times)},
        {"path": "cufinufft_measured_pair", "median_s": cu_pair_s, "times_s": " ".join(f"{v:.9g}" for v in cu_pair_times)},
    ]
    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "rows": rows})
    if args.summary_md:
        write_summary(args.summary_md, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark the public aIDT measured-data adapter on prepared GPU and cuFINUFFT GPU operators.")
    p.add_argument("--contract", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--cap-radial", type=int, default=32)
    p.add_argument("--cap-phi", type=int, default=128)
    p.add_argument("--detector-na", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fft-norm", choices=["none", "ortho"], default="ortho")
    p.add_argument("--interpolation-order", type=int, choices=[0, 1, 3], default=1)
    p.add_argument("--no-residual-normalize", action="store_true")
    p.add_argument("--n-beta", type=int, default=256)
    p.add_argument("--n-r", type=int, default=12)
    p.add_argument("--n-z", type=int, default=11)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.93636473255199)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    p.add_argument("--native-prepared-plan-mode", choices=["auto", "direct", "gathered", "gathered-zmajor"], default="auto")
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_adapter_benchmark.json")
    p.add_argument("--csv", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_adapter_benchmark.csv")
    p.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_adapter_benchmark.md")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
