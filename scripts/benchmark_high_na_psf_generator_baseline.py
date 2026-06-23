from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
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

from benchmark_high_na_debye_wolf import gauss_theta_grid  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    make_pupil_batch,
    resolve_device,
    timed_torch,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_pupil_jones,
)


def version_or_missing(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


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


def build_ours_inputs(args: argparse.Namespace, torch: Any, device: Any):
    theta, theta_weights = gauss_theta_grid(args.ntheta, args.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, args.nphi, endpoint=False, dtype=float)
    rho_axis = np.linspace(0.0, args.rho_max, args.nrho, dtype=float)
    psi_axis = np.linspace(0.0, 2.0 * np.pi, args.npsi, endpoint=False, dtype=float)
    if args.nz == 1:
        z_axis = np.array([0.0], dtype=float)
    else:
        z_axis = np.linspace(-args.z_max, args.z_max, args.nz, dtype=float)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    base_pupil = vectorial_pupil_jones(
        args.pupil_case,
        theta,
        phi,
        theta_max=args.theta_max,
        strength=args.aberration_strength,
        vortex_charge=args.vortex_charge,
    )
    pupil_batch = make_pupil_batch(
        base_pupil,
        theta,
        phi,
        theta_max=args.theta_max,
        batch_size=args.batch_size,
        phase_strength=args.batch_phase_strength,
    )
    plan = TorchSeparableHarmonicDebyeWolfPlan.build(
        torch=torch,
        nphi=args.nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=args.k,
        h_cutoff=args.h_cutoff,
        device=device,
        dtype=args.dtype,
        basis_mode=args.basis_mode,
        contract_mode=args.contract_mode,
    )
    return plan, plan.as_tensor(pupil_batch), plan.as_tensor(mixing)


def run_case(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = import_torch()
    if torch is None:
        return [{"status": "skipped", "skip_reason": "torch is not installed"}]
    if importlib.util.find_spec("psf_generator") is None:
        return [{"status": "skipped", "skip_reason": "psf_generator is not installed"}]
    from psf_generator.propagators import VectorialCartesianPropagator

    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    psf = VectorialCartesianPropagator(
        n_pix_pupil=args.psf_n_pix_pupil,
        n_pix_psf=args.psf_n_pix_psf,
        device=str(device),
        wavelength=args.wavelength_nm,
        na=args.na,
        pix_size=args.psf_pix_size_nm,
        defocus_step=args.psf_defocus_step_nm,
        n_defocus=args.psf_n_defocus,
        e0x=args.e0x,
        e0y=args.e0y,
        apod_factor=args.psf_apod_factor,
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

    ours_plan, pupil_batch_t, mixing_t = build_ours_inputs(args, torch, device)
    ours_value, ours_hot_s, ours_times = timed_torch(
        torch,
        device,
        lambda: ours_plan.evaluate_vectorial_batch(pupil_batch_t, mixing_t),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    del ours_value
    psf_shape = tuple(int(dim) for dim in psf_value.shape)
    psf_targets = int(args.psf_n_defocus * args.psf_n_pix_psf * args.psf_n_pix_psf)
    psf_output_elements = int(np.prod(psf_shape))
    ours_targets = int(args.nrho * args.npsi * args.nz)
    ours_output_elements = int(args.batch_size * 3 * ours_targets)
    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return [
        {
            "status": "ok",
            "baseline": "psf_generator_vectorial_cartesian",
            "device": str(device),
            "device_name": device_name(torch, device),
            "torch_version": version_or_missing("torch"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "psf_generator_version": version_or_missing("psf-generator"),
            "dtype": args.dtype,
            "batch_size_ours": args.batch_size,
            "psf_n_pix_pupil": args.psf_n_pix_pupil,
            "psf_n_pix_psf": args.psf_n_pix_psf,
            "psf_n_defocus": args.psf_n_defocus,
            "psf_targets": psf_targets,
            "psf_output_elements": psf_output_elements,
            "psf_output_shape": str(psf_shape),
            "psf_hot_s": float(psf_hot_s),
            "ours_nrho": args.nrho,
            "ours_npsi": args.npsi,
            "ours_nz": args.nz,
            "ours_targets_per_mask": ours_targets,
            "ours_total_output_fields": int(args.batch_size * 3),
            "ours_output_elements": ours_output_elements,
            "ours_h_cutoff": args.h_cutoff,
            "ours_basis_mode": ours_plan.basis_mode,
            "ours_contract_mode": ours_plan.contract_mode,
            "ours_used_modes": int(ours_plan.used_modes),
            "ours_basis_mib": float(ours_plan.basis_mib),
            "ours_hot_s": float(ours_hot_s),
            "wall_time_ratio_psf_generator_cartesian_to_ours_structured": float(psf_hot_s / ours_hot_s),
            "gpu_peak_allocated_mib": peak_mib,
            "psf_times_s": " ".join(f"{value:.9g}" for value in psf_times),
            "ours_times_s": " ".join(f"{value:.9g}" for value in ours_times),
            "note": "timing only; grids and pupil parameterizations are not matched",
        }
    ]


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
        "# High-NA psf-generator GPU baseline",
        "",
        "This benchmark records the external `psf-generator` VectorialCartesianPropagator timing next to the local Torch separable High-NA timing on the same CUDA device.",
        "It is a throughput baseline, not a matched accuracy comparison: psf-generator evaluates a dense Cartesian PSF grid while the local method evaluates a structured cylindrical grid.",
        "",
        "## Results",
        "",
        "| baseline | device | PSF spatial targets | PSF elements | PSF hot s | ours targets/mask | ours elements | ours hot s | wall-time ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('status')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            lines.append("")
            lines.append(f"Skip reason: `{row.get('skip_reason')}`")
            continue
        lines.append(
            "| {baseline} | {device} | {psf_targets} | {psf_elements} | {psf_hot} | {ours_targets} | {ours_elements} | {ours_hot} | {ratio} |".format(
                baseline=row["baseline"],
                device=row["device_name"],
                psf_targets=row["psf_targets"],
                psf_elements=row["psf_output_elements"],
                psf_hot=fmt(row["psf_hot_s"]),
                ours_targets=row["ours_targets_per_mask"],
                ours_elements=row["ours_output_elements"],
                ours_hot=fmt(row["ours_hot_s"]),
                ratio=fmt(row["wall_time_ratio_psf_generator_cartesian_to_ours_structured"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This row does not compare identical pupil functions or identical target coordinates.",
            "- Use it as an external domain-package timing anchor before building a fully matched adapter.",
            "- `psf-generator` is the right package family for a future matched dense-Cartesian GPU baseline because it exposes vectorial Cartesian and spherical propagators built on PyTorch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark psf-generator VectorialCartesianPropagator against the local Torch separable High-NA path."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_psf_generator_baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--na", type=float, default=0.95)
    parser.add_argument("--refractive-index", type=float, default=1.0)
    parser.add_argument("--e0x", type=float, default=1.0)
    parser.add_argument("--e0y", type=float, default=0.0)
    parser.add_argument("--psf-n-pix-pupil", type=int, default=128)
    parser.add_argument("--psf-n-pix-psf", type=int, default=64)
    parser.add_argument("--psf-n-defocus", type=int, default=5)
    parser.add_argument("--psf-pix-size-nm", type=float, default=50.0)
    parser.add_argument("--psf-defocus-step-nm", type=float, default=100.0)
    parser.add_argument("--psf-apod-factor", action="store_true")
    parser.add_argument("--ntheta", type=int, default=28)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--nrho", type=int, default=16)
    parser.add_argument("--npsi", type=int, default=64)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--rho-max", type=float, default=2.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=float(np.arcsin(0.95)))
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--h-cutoff", type=int, default=23)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="fused")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="matmul")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-phase-strength", type=float, default=0.25)
    parser.add_argument("--pupil-case", choices=["x_vortex", "mixed_jones", "radial_donut", "azimuthal_donut"], default="mixed_jones")
    parser.add_argument("--aberration-strength", type=float, default=0.45)
    parser.add_argument("--vortex-charge", type=int, default=8)
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
