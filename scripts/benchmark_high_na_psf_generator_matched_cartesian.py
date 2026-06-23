from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
(ROOT / ".matplotlib_cache").mkdir(exist_ok=True)

from benchmark_high_na_cartesian_interpolation_adapter import build_interpolation_map  # noqa: E402
from benchmark_high_na_debye_wolf import gauss_theta_grid, relative_l2  # noqa: E402
from benchmark_high_na_gpu_dense_baseline import TorchDenseDirectVectorialDebyeWolf  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    package_version,
    resolve_device,
    timed_torch,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import richards_wolf_jones_matrix  # noqa: E402


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def timed_cuda(torch: Any, device: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        value = func()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def cartesian_targets_from_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="xy")
    return xx.ravel(), yy.ravel(), zz.ravel()


def as_zcyx(field_bcn: np.ndarray, *, batch: int, ny: int, nx: int, nz: int) -> np.ndarray:
    field = field_bcn.reshape(batch, 3, ny, nx, nz)[0]
    return np.moveaxis(field, -1, 0)


def intensity_zcyx(field: np.ndarray) -> np.ndarray:
    if field.ndim != 4:
        raise ValueError("field must have shape (z, component, y, x)")
    return np.sum(np.abs(field) ** 2, axis=1)


def scale_fit_l2(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    denom = float(np.dot(cand, cand))
    if denom == 0.0:
        return float("inf"), 0.0
    scale = float(np.dot(ref, cand) / denom)
    err = float(np.linalg.norm(scale * cand - ref) / max(np.linalg.norm(ref), 1e-300))
    return err, scale


def transformed_intensities(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    transforms: list[tuple[str, np.ndarray]] = []
    xy_ops = {
        "identity": values,
        "flip_y": values[:, ::-1, :],
        "flip_x": values[:, :, ::-1],
        "flip_xy": values[:, ::-1, ::-1],
        "transpose": np.swapaxes(values, -1, -2),
        "transpose_flip_y": np.swapaxes(values, -1, -2)[:, ::-1, :],
        "transpose_flip_x": np.swapaxes(values, -1, -2)[:, :, ::-1],
        "transpose_flip_xy": np.swapaxes(values, -1, -2)[:, ::-1, ::-1],
    }
    for name, item in xy_ops.items():
        transforms.append((name, item))
        transforms.append((name + "_zflip", item[::-1, :, :]))
    return transforms


def best_intensity_shape(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    best: tuple[float, float, str, np.ndarray] | None = None
    for name, transformed in transformed_intensities(candidate):
        err, scale = scale_fit_l2(reference, transformed)
        if best is None or err < best[0]:
            best = (err, scale, name, transformed)
    if best is None:
        raise RuntimeError("no transforms evaluated")
    err, scale, name, transformed = best
    corr = pearson(reference, transformed)
    return {
        "scale_fit_intensity_l2": float(err),
        "scale_fit_intensity_scale": float(scale),
        "pearson": float(corr),
        "orientation": name,
    }


def pearson(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    ref = ref - float(np.mean(ref))
    cand = cand - float(np.mean(cand))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(ref, cand) / denom)


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
        "# High-NA psf-generator matched Cartesian sanity benchmark",
        "",
        "This benchmark uses a deliberately simple condition that can be matched across implementations: clear pupil, linear x polarization, no apodization, homogeneous refractive index, and identical Cartesian PSF stack dimensions.",
        "",
        "The package field is `psf_generator.propagators.VectorialCartesianPropagator`. The local references are a dense direct vectorial Debye-Wolf CUDA quadrature and the native cylindrical solver followed by trilinear interpolation to the same Cartesian target stack.",
        "",
        "Complex field components are not compared directly because package conventions can differ by global/component phase and image-axis orientation. The reported accuracy metric is scale-fit intensity-shape L2 after the best discrete XY/Z orientation transform.",
        "",
        "## Config",
        "",
        f"- GPU/device: `{config.get('device_name', config['device'])}`",
        f"- psf_generator: `{config['psf_generator_version']}`",
        f"- stack: `{config['n_pix_psf']} x {config['n_pix_psf']} x {config['n_defocus']}`",
        f"- pupil samples: package `{config['n_pix_pupil']}` Cartesian, local `{config['ntheta']} x {config['nphi']}` polar",
        f"- wavelength: `{config['wavelength_nm']}` nm",
        f"- NA / n: `{config['na']}` / `{config['refractive_index']}`",
        "",
        "## Results",
        "",
        "| oversample | cyl grid | psf ms | dense ms | eval+interp ms | psf/adapter | dense/adapter | dense-vs-psf L2 | adapter-vs-psf L2 | adapter-vs-dense L2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {os} | {grid} | {psf} | {dense} | {total} | {rpsf} | {rdense} | {ldp} | {lap} | {lad} |".format(
                os=row["oversample"],
                grid=f"{row['nrho']}x{row['npsi']}x{row['nz']}",
                psf=fmt(1e3 * row["psf_hot_s"]),
                dense=fmt(1e3 * row["dense_cartesian_hot_s"]),
                total=fmt(1e3 * row["adapter_eval_plus_interp_hot_s"]),
                rpsf=fmt(row["speedup_psf_vs_adapter"]),
                rdense=fmt(row["speedup_dense_vs_adapter"]),
                ldp=fmt(row["dense_vs_psf_intensity_l2"]),
                lap=fmt(row["adapter_vs_psf_intensity_l2"]),
                lad=fmt(row["adapter_vs_dense_intensity_l2"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a Cartesian stress test, not the native target regime for the local method.",
            "- The strongest manuscript claim should still be made on cylindrical/polar or ROI workloads.",
            "- This row only shows that even after paying the cylindrical-to-Cartesian interpolation cost, the method remains competitive under a simple package-matchable condition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = import_torch()
    if torch is None:
        config = vars(args).copy()
        return config, [{"status": "skipped", "skip_reason": "torch is not installed"}]
    if importlib.util.find_spec("psf_generator") is None:
        config = vars(args).copy()
        return config, [{"status": "skipped", "skip_reason": "psf_generator is not installed"}]

    from psf_generator.propagators import VectorialCartesianPropagator

    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    psf = VectorialCartesianPropagator(
        n_pix_pupil=args.n_pix_pupil,
        n_pix_psf=args.n_pix_psf,
        device=str(device),
        wavelength=args.wavelength_nm,
        na=args.na,
        pix_size=args.pix_size_nm,
        defocus_step=args.defocus_step_nm,
        n_defocus=args.n_defocus,
        e0x=args.e0x,
        e0y=args.e0y,
        apod_factor=False,
        gibson_lanni=False,
        n_i=args.refractive_index,
        n_i0=args.refractive_index,
        n_s=args.refractive_index,
    )
    psf_value, psf_hot_s, psf_times = timed_cuda(
        torch,
        device,
        psf.compute_focus_field,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    psf_zcyx = psf_value.detach().cpu().numpy()
    psf_intensity = intensity_zcyx(psf_zcyx)

    theta_max = float(math.asin(args.na / args.refractive_index))
    theta, theta_weights = gauss_theta_grid(args.ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, args.nphi, endpoint=False, dtype=float)
    k = 2.0 * np.pi * args.refractive_index / args.wavelength_nm
    x_axis = np.linspace(-psf.fov / 2.0, psf.fov / 2.0, args.n_pix_psf, dtype=float)
    y_axis = np.linspace(-psf.fov / 2.0, psf.fov / 2.0, args.n_pix_psf, dtype=float)
    z_axis = np.linspace(float(psf.defocus_min), float(psf.defocus_max), args.n_defocus, dtype=float)
    x, y, z = cartesian_targets_from_axes(x_axis, y_axis, z_axis)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="none")
    pupil = np.zeros((1, 2, args.ntheta, args.nphi), dtype=np.complex128)
    pupil[0, 0] = complex(args.e0x)
    pupil[0, 1] = complex(args.e0y)

    dense = TorchDenseDirectVectorialDebyeWolf(
        torch=torch,
        theta=theta,
        theta_weights=theta_weights,
        phi=phi,
        x=x,
        y=y,
        z=z,
        k=k,
        mixing=mixing,
        device=device,
        dtype=args.dtype,
        chunk_targets=args.chunk_targets,
    )
    pupil_t = torch.as_tensor(pupil, dtype=dense.complex_dtype, device=device)
    dense_value, dense_hot_s, dense_times = timed_torch(
        torch,
        device,
        lambda: dense.forward(pupil_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    dense_zcyx = as_zcyx(
        to_numpy(torch, device, dense_value),
        batch=1,
        ny=args.n_pix_psf,
        nx=args.n_pix_psf,
        nz=args.n_defocus,
    )
    dense_intensity = intensity_zcyx(dense_zcyx)
    dense_vs_psf = best_intensity_shape(psf_intensity, dense_intensity)

    rows: list[dict[str, Any]] = []
    for oversample in parse_int_list(args.oversamples):
        nrho = int(args.base_nrho * oversample)
        npsi = int(args.base_npsi * oversample)
        rho_max = float(psf.fov / math.sqrt(2.0) * args.rho_margin)
        rho_axis = np.linspace(0.0, rho_max, nrho, dtype=float)
        psi_axis = np.linspace(0.0, 2.0 * np.pi, npsi, endpoint=False, dtype=float)
        h_cutoff = int(
            min(
                args.nphi // 2,
                math.ceil(k * rho_max * math.sin(theta_max)) + int(args.h_margin),
            )
        )

        setup_start = time.perf_counter()
        plan = TorchSeparableHarmonicDebyeWolfPlan.build(
            torch=torch,
            nphi=args.nphi,
            theta=theta,
            theta_weights=theta_weights,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            k=k,
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
        cyl_field = cyl_flat.reshape(1, 3, nrho, npsi, args.n_defocus)
        interp_value, interp_s, interp_times = timed_torch(
            torch,
            device,
            lambda: interp.forward(cyl_field),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        del interp_value
        adapter_value, adapter_total_s, adapter_total_times = timed_torch(
            torch,
            device,
            lambda: interp.forward(
                plan.evaluate_vectorial_batch(pupil_t, mixing_t).reshape(
                    1,
                    3,
                    nrho,
                    npsi,
                    args.n_defocus,
                )
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        adapter_zcyx = as_zcyx(
            to_numpy(torch, device, adapter_value),
            batch=1,
            ny=args.n_pix_psf,
            nx=args.n_pix_psf,
            nz=args.n_defocus,
        )
        adapter_intensity = intensity_zcyx(adapter_zcyx)
        adapter_vs_psf = best_intensity_shape(psf_intensity, adapter_intensity)
        adapter_vs_dense_l2, adapter_vs_dense_scale = scale_fit_l2(dense_intensity, adapter_intensity)
        rows.append(
            {
                "status": "ok",
                "device": str(device),
                "device_name": device_name(torch, device),
                "torch_version": package_version("torch"),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "psf_generator_version": version_or_missing("psf-generator"),
                "dtype": args.dtype,
                "n_pix_pupil": int(args.n_pix_pupil),
                "n_pix_psf": int(args.n_pix_psf),
                "n_defocus": int(args.n_defocus),
                "cartesian_targets": int(x.size),
                "ntheta": int(args.ntheta),
                "nphi": int(args.nphi),
                "oversample": int(oversample),
                "nrho": int(nrho),
                "npsi": int(npsi),
                "nz": int(args.n_defocus),
                "cylindrical_targets": int(nrho * npsi * args.n_defocus),
                "h_cutoff": int(h_cutoff),
                "used_modes": int(plan.used_modes),
                "basis_mib": float(plan.basis_mib),
                "interpolation_map_mib": float(interp.map_mib),
                "setup_s": float(setup_s),
                "psf_hot_s": float(psf_hot_s),
                "dense_cartesian_hot_s": float(dense_hot_s),
                "adapter_eval_hot_s": float(eval_s),
                "adapter_interp_hot_s": float(interp_s),
                "adapter_eval_plus_interp_hot_s": float(adapter_total_s),
                "speedup_psf_vs_adapter": float(psf_hot_s / adapter_total_s),
                "speedup_dense_vs_adapter": float(dense_hot_s / adapter_total_s),
                "dense_vs_psf_intensity_l2": float(dense_vs_psf["scale_fit_intensity_l2"]),
                "dense_vs_psf_scale": float(dense_vs_psf["scale_fit_intensity_scale"]),
                "dense_vs_psf_pearson": float(dense_vs_psf["pearson"]),
                "dense_vs_psf_orientation": str(dense_vs_psf["orientation"]),
                "adapter_vs_psf_intensity_l2": float(adapter_vs_psf["scale_fit_intensity_l2"]),
                "adapter_vs_psf_scale": float(adapter_vs_psf["scale_fit_intensity_scale"]),
                "adapter_vs_psf_pearson": float(adapter_vs_psf["pearson"]),
                "adapter_vs_psf_orientation": str(adapter_vs_psf["orientation"]),
                "adapter_vs_dense_intensity_l2": float(adapter_vs_dense_l2),
                "adapter_vs_dense_scale": float(adapter_vs_dense_scale),
                "psf_times_s": " ".join(f"{value:.9g}" for value in psf_times),
                "dense_times_s": " ".join(f"{value:.9g}" for value in dense_times),
                "adapter_eval_times_s": " ".join(f"{value:.9g}" for value in eval_times),
                "adapter_interp_times_s": " ".join(f"{value:.9g}" for value in interp_times),
                "adapter_eval_plus_interp_times_s": " ".join(f"{value:.9g}" for value in adapter_total_times),
                "gpu_peak_allocated_mib": None
                if device.type != "cuda"
                else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            }
        )
        del cyl_flat, cyl_field, adapter_value

    config = vars(args).copy()
    config.update(
        {
            "device_name": device_name(torch, device),
            "psf_generator_version": version_or_missing("psf-generator"),
            "fov_nm": float(psf.fov),
            "defocus_min_nm": float(psf.defocus_min),
            "defocus_max_nm": float(psf.defocus_max),
        }
    )
    return config, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched clear-pupil Cartesian sanity comparison for psf-generator, local dense direct, and cylindrical interpolation adapter."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_psf_generator_matched_cartesian")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--n-pix-pupil", type=int, default=128)
    parser.add_argument("--n-pix-psf", type=int, default=64)
    parser.add_argument("--n-defocus", type=int, default=5)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--na", type=float, default=0.95)
    parser.add_argument("--refractive-index", type=float, default=1.0)
    parser.add_argument("--pix-size-nm", type=float, default=50.0)
    parser.add_argument("--defocus-step-nm", type=float, default=100.0)
    parser.add_argument("--e0x", type=float, default=1.0)
    parser.add_argument("--e0y", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--nphi", type=int, default=256)
    parser.add_argument("--base-nrho", type=int, default=64)
    parser.add_argument("--base-npsi", type=int, default=192)
    parser.add_argument("--oversamples", default="1,2,4")
    parser.add_argument("--rho-margin", type=float, default=1.001)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="fused")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="matmul")
    parser.add_argument("--chunk-targets", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, rows = run_case(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
