from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake.odt_measured_contract import (  # noqa: E402
    load_odt_measured_contract,
    validate_odt_measured_contract,
)


@dataclass(frozen=True)
class AidtPreprocessedData:
    data: np.ndarray
    frequency_x: np.ndarray
    frequency_y: np.ndarray
    crop_size: int
    crop_origin_yx: tuple[int, int]
    background_mode: str
    flatfield_mode: str
    frame_scale_mode: str
    preprocessing_stats: dict[str, Any]


@dataclass(frozen=True)
class TransferAccumulation:
    sum_ptf: np.ndarray
    sum_atf: np.ndarray
    conj_ptf_image: np.ndarray
    conj_atf_image: np.ndarray
    conj_ptf_atf: np.ndarray
    conj_atf_ptf: np.ndarray
    stats: dict[str, Any]


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def centered_frequency_axis(n: int, pixel_size_um: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(int(n), d=float(pixel_size_um))).astype(np.float64)


def centered_crop_stack(data: np.ndarray, crop_size: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    if crop_size is None or crop_size <= 0:
        return np.ascontiguousarray(data), (0, 0)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D stack, got shape {data.shape}")
    if crop_size > data.shape[1] or crop_size > data.shape[2]:
        raise ValueError(f"crop_size {crop_size} exceeds stack shape {data.shape}")
    y0 = (data.shape[1] - crop_size) // 2
    x0 = (data.shape[2] - crop_size) // 2
    return np.ascontiguousarray(data[:, y0 : y0 + crop_size, x0 : x0 + crop_size]), (int(y0), int(x0))


def centered_fft2(image: np.ndarray, *, norm: str) -> np.ndarray:
    fft_norm = None if norm == "none" else norm
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm=fft_norm))


def centered_ifft2(spectrum: np.ndarray, *, norm: str, axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    fft_norm = None if norm == "none" else norm
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(spectrum, axes=axes), axes=axes, norm=fft_norm),
        axes=axes,
    )


def preprocess_aidt_data(
    measured: Any,
    *,
    crop_size: int | None,
    background_mode: str,
    flatfield_mode: str,
    frame_scale_mode: str,
) -> AidtPreprocessedData:
    raw = np.asarray(measured["data"], dtype=np.float32)
    cropped, crop_origin_yx = centered_crop_stack(raw, crop_size)
    data = cropped.astype(np.float32, copy=True)
    raw_stats = {
        "raw_min": float(np.min(data)),
        "raw_max": float(np.max(data)),
        "raw_mean": float(np.mean(data)),
        "raw_std": float(np.std(data)),
    }

    if flatfield_mode == "global_mean":
        flatfield = np.mean(np.maximum(raw, 1e-6), axis=0)
        flatfield_crop, _ = centered_crop_stack(flatfield[None, :, :], crop_size)
        flat = flatfield_crop[0].astype(np.float32)
        flat = flat / max(float(np.mean(flat)), 1e-6)
        data = data / np.maximum(flat[None, :, :], 1e-3)
    elif flatfield_mode != "none":
        raise ValueError(f"unsupported flatfield mode {flatfield_mode!r}")

    if background_mode == "frame_mean":
        data = data - np.mean(data, axis=(1, 2), keepdims=True)
    elif background_mode == "global_mean":
        data = data - float(np.mean(data))
    elif background_mode == "edge_median":
        edge = np.concatenate(
            [
                data[:, :4, :].reshape(data.shape[0], -1),
                data[:, -4:, :].reshape(data.shape[0], -1),
                data[:, :, :4].reshape(data.shape[0], -1),
                data[:, :, -4:].reshape(data.shape[0], -1),
            ],
            axis=1,
        )
        data = data - np.median(edge, axis=1)[:, None, None].astype(np.float32)
    elif background_mode != "none":
        raise ValueError(f"unsupported background mode {background_mode!r}")

    if frame_scale_mode == "rms":
        scale = np.sqrt(np.mean(data * data, axis=(1, 2), keepdims=True))
        data = data / np.maximum(scale, 1e-6)
    elif frame_scale_mode == "none":
        pass
    else:
        raise ValueError(f"unsupported frame scale mode {frame_scale_mode!r}")

    pixel_size = np.asarray(measured["detector_pixel_size"], dtype=np.float64)
    if pixel_size.shape != (2,):
        raise ValueError(f"detector_pixel_size must have shape (2,), got {pixel_size.shape}")
    frequency_x = centered_frequency_axis(data.shape[2], float(pixel_size[0]))
    frequency_y = centered_frequency_axis(data.shape[1], float(pixel_size[1]))
    preprocessing_stats = {
        **raw_stats,
        "processed_min": float(np.min(data)),
        "processed_max": float(np.max(data)),
        "processed_mean": float(np.mean(data)),
        "processed_std": float(np.std(data)),
        "crop_origin_yx": [int(crop_origin_yx[0]), int(crop_origin_yx[1])],
    }
    return AidtPreprocessedData(
        data=np.ascontiguousarray(data.astype(np.float32)),
        frequency_x=frequency_x,
        frequency_y=frequency_y,
        crop_size=int(data.shape[1]),
        crop_origin_yx=crop_origin_yx,
        background_mode=background_mode,
        flatfield_mode=flatfield_mode,
        frame_scale_mode=frame_scale_mode,
        preprocessing_stats=preprocessing_stats,
    )


def transfer_functions_for_source(
    *,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    source_na_xy: np.ndarray,
    wavelength_um: float,
    medium_index: float,
    objective_na: float,
    depth_values_um: np.ndarray,
    dz_um: float,
    evanescent_eps: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fx = frequency_x[None, :]
    fy = frequency_y[:, None]
    source_fx = float(source_na_xy[0]) / float(wavelength_um)
    source_fy = float(source_na_xy[1]) / float(wavelength_um)
    k0 = 2.0 * math.pi / float(wavelength_um)
    k_medium = k0 * float(medium_index)
    max_frequency = float(objective_na) / float(wavelength_um)

    def axial_and_green(delta_fx: np.ndarray, delta_fy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radial2 = delta_fx * delta_fx + delta_fy * delta_fy
        inside = 1.0 - float(wavelength_um) ** 2 * radial2
        valid = (inside > float(evanescent_eps)) & (radial2 <= max_frequency * max_frequency)
        axial = np.zeros_like(inside, dtype=np.float64)
        axial[valid] = np.sqrt(inside[valid])
        green = np.zeros_like(inside, dtype=np.float64)
        green[valid] = 1.0 / (k_medium * axial[valid])
        return axial, green, valid

    uv1, green1, pupil1 = axial_and_green(fx - source_fx, fy - source_fy)
    uv2, green2, pupil2 = axial_and_green(fx + source_fx, fy + source_fy)
    source_inside = 1.0 - float(wavelength_um) ** 2 * (source_fx * source_fx + source_fy * source_fy)
    if source_inside <= float(evanescent_eps):
        raise ValueError(f"source NA is evanescent or grazing: {source_na_xy}")
    source_axial = math.sqrt(source_inside)
    z = depth_values_um[None, None, :].astype(np.float64)
    phase1 = k_medium * z * (uv1[:, :, None] - source_axial)
    phase2 = k_medium * z * (uv2[:, :, None] - source_axial)
    green1_3d = green1[:, :, None]
    green2_3d = green2[:, :, None]
    pupil1_3d = pupil1[:, :, None].astype(np.float64)
    pupil2_3d = pupil2[:, :, None].astype(np.float64)

    ptf = (
        pupil1_3d * np.sin(phase1) * green1_3d
        + pupil2_3d * np.sin(phase2) * green2_3d
    ) + 1j * (
        pupil1_3d * np.cos(phase1) * green1_3d
        - pupil2_3d * np.cos(phase2) * green2_3d
    )
    atf = -(
        pupil1_3d * np.cos(phase1) * green1_3d
        + pupil2_3d * np.cos(phase2) * green2_3d
    ) + 1j * (
        pupil1_3d * np.sin(phase1) * green1_3d
        - pupil2_3d * np.sin(phase2) * green2_3d
    )
    scale = 0.5 * float(dz_um) * k0 * k0
    ptf = (scale * ptf).astype(dtype, copy=False)
    atf = (scale * atf).astype(dtype, copy=False)
    stats = {
        "source_fx": source_fx,
        "source_fy": source_fy,
        "valid_forward_fraction": float(np.mean(pupil1)),
        "valid_conjugate_fraction": float(np.mean(pupil2)),
        "ptf_abs_max": float(np.max(np.abs(ptf))),
        "atf_abs_max": float(np.max(np.abs(atf))),
    }
    return ptf, atf, stats


def accumulate_transfer_system(
    *,
    data: np.ndarray,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    source_na_xy: np.ndarray,
    wavelength_um: float,
    medium_index: float,
    objective_na: float,
    depth_values_um: np.ndarray,
    dz_um: float,
    fft_norm: str,
    evanescent_eps: float,
    complex_dtype: np.dtype,
) -> TransferAccumulation:
    if data.shape[0] != source_na_xy.shape[0]:
        raise ValueError(f"data/source count mismatch: {data.shape[0]} vs {source_na_xy.shape[0]}")
    h, w = data.shape[1:]
    nz = int(depth_values_um.size)
    accum_shape = (h, w, nz)
    real_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    sum_ptf = np.zeros(accum_shape, dtype=real_dtype)
    sum_atf = np.zeros(accum_shape, dtype=real_dtype)
    conj_ptf_image = np.zeros(accum_shape, dtype=complex_dtype)
    conj_atf_image = np.zeros(accum_shape, dtype=complex_dtype)
    conj_ptf_atf = np.zeros(accum_shape, dtype=complex_dtype)
    conj_atf_ptf = np.zeros(accum_shape, dtype=complex_dtype)
    source_stats: list[dict[str, Any]] = []
    fft_abs_norms: list[float] = []

    start = time.perf_counter()
    for frame_index in range(data.shape[0]):
        image_fft = centered_fft2(data[frame_index], norm=fft_norm).astype(complex_dtype, copy=False)
        fft_abs_norms.append(float(np.linalg.norm(image_fft.astype(np.complex128))))
        ptf, atf, stats = transfer_functions_for_source(
            frequency_x=frequency_x,
            frequency_y=frequency_y,
            source_na_xy=source_na_xy[frame_index],
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            objective_na=objective_na,
            depth_values_um=depth_values_um,
            dz_um=dz_um,
            evanescent_eps=evanescent_eps,
            dtype=complex_dtype,
        )
        source_stats.append({"frame_index": int(frame_index), **stats})
        sum_ptf += np.real(np.conj(ptf) * ptf)
        sum_atf += np.real(np.conj(atf) * atf)
        conj_ptf_image += np.conj(ptf) * image_fft[:, :, None]
        conj_atf_image += np.conj(atf) * image_fft[:, :, None]
        conj_ptf_atf += np.conj(ptf) * atf
        conj_atf_ptf += np.conj(atf) * ptf

    stats = {
        "accumulation_s": float(time.perf_counter() - start),
        "source_stats": source_stats,
        "fft_abs_norm_min": float(np.min(fft_abs_norms)),
        "fft_abs_norm_max": float(np.max(fft_abs_norms)),
        "sum_ptf_max": float(np.max(sum_ptf)),
        "sum_atf_max": float(np.max(sum_atf)),
        "sum_ptf_nonzero_fraction": float(np.mean(sum_ptf > 0)),
        "sum_atf_nonzero_fraction": float(np.mean(sum_atf > 0)),
    }
    return TransferAccumulation(
        sum_ptf=sum_ptf,
        sum_atf=sum_atf,
        conj_ptf_image=conj_ptf_image,
        conj_atf_image=conj_atf_image,
        conj_ptf_atf=conj_ptf_atf,
        conj_atf_ptf=conj_atf_ptf,
        stats=stats,
    )


def solve_transfer_system(
    accumulation: TransferAccumulation,
    *,
    alpha: float,
    beta: float,
    fft_norm: str,
    medium_index: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    sum_ptf = accumulation.sum_ptf
    sum_atf = accumulation.sum_atf
    denom = (sum_ptf + float(alpha)) * (sum_atf + float(beta)) - (
        accumulation.conj_ptf_atf * accumulation.conj_atf_ptf
    )
    denom_abs = np.abs(denom)
    denom_floor = max(float(np.max(denom_abs)) * 1e-12, 1e-30)
    denom = np.where(denom_abs > denom_floor, denom, denom_floor + 0j)
    v_re_freq = (
        (sum_atf + float(beta)) * accumulation.conj_ptf_image
        - accumulation.conj_ptf_atf * accumulation.conj_atf_image
    ) / denom
    v_im_freq = (
        (sum_ptf + float(alpha)) * accumulation.conj_atf_image
        - accumulation.conj_atf_ptf * accumulation.conj_ptf_image
    ) / denom
    v_re = np.real(centered_ifft2(v_re_freq, norm=fft_norm, axes=(0, 1))).astype(np.float32)
    v_im = np.real(centered_ifft2(v_im_freq, norm=fft_norm, axes=(0, 1))).astype(np.float32)
    n2_plus_v = float(medium_index) ** 2 + v_re
    n_re = np.sqrt(np.maximum((n2_plus_v + np.sqrt(np.maximum(n2_plus_v * n2_plus_v + v_im * v_im, 0.0))) / 2.0, 0.0))
    n_im = np.divide(v_im, 2.0 * np.maximum(n_re, 1e-12))
    stats = {
        "solve_s": float(time.perf_counter() - start),
        "denom_abs_min": float(np.min(denom_abs)),
        "denom_abs_max": float(np.max(denom_abs)),
        "denom_floor": float(denom_floor),
        "v_re_min": float(np.min(v_re)),
        "v_re_max": float(np.max(v_re)),
        "v_im_min": float(np.min(v_im)),
        "v_im_max": float(np.max(v_im)),
        "n_re_min": float(np.min(n_re)),
        "n_re_max": float(np.max(n_re)),
        "n_im_min": float(np.min(n_im)),
        "n_im_max": float(np.max(n_im)),
    }
    return v_re, v_im, n_re.astype(np.float32), n_im.astype(np.float32), stats


def make_depth_values(depth_min_um: float, depth_max_um: float, depth_step_um: float) -> np.ndarray:
    if depth_step_um <= 0:
        raise ValueError("depth_step_um must be positive")
    count = int(round((depth_max_um - depth_min_um) / depth_step_um)) + 1
    if count <= 0:
        raise ValueError("depth range is empty")
    values = depth_min_um + depth_step_um * np.arange(count, dtype=np.float64)
    if values[-1] > depth_max_um + 1e-9:
        values = values[values <= depth_max_um + 1e-9]
    return values.astype(np.float64)


def write_npz(
    path: Path,
    *,
    v_re: np.ndarray,
    v_im: np.ndarray,
    n_re: np.ndarray,
    n_im: np.ndarray,
    depth_values_um: np.ndarray,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        v_re=v_re,
        v_im=v_im,
        n_re=n_re,
        n_im=n_im,
        depth_values_um=depth_values_um,
        frequency_x=frequency_x,
        frequency_y=frequency_y,
        summary_json=np.asarray(json.dumps(summary, default=json_default)),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Public aIDT transfer-function reconstruction",
        "",
        "This run applies the public aIDT pupil/phase transfer-function equations to the converted Diatom I measured-intensity contract.",
        "It is a physics-aware reconstruction smoke test, not yet a prepared-operator speed benchmark.",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key in (
        "contract",
        "crop_size",
        "depth_count",
        "depth_min_um",
        "depth_max_um",
        "depth_step_um",
        "background_mode",
        "flatfield_mode",
        "frame_scale_mode",
        "fft_norm",
        "alpha",
        "beta",
        "complex_dtype",
    ):
        lines.append(f"| `{key}` | `{summary.get(key)}` |")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| stage | seconds |",
            "| --- | ---: |",
            f"| preprocessing | {summary['preprocessing_s']:.6f} |",
            f"| transfer accumulation | {summary['accumulation_s']:.6f} |",
            f"| solve and RI conversion | {summary['solve_s']:.6f} |",
            f"| total | {summary['total_s']:.6f} |",
            "",
            "## Output Ranges",
            "",
            "| quantity | min | max |",
            "| --- | ---: | ---: |",
            f"| `V_re` | {summary['v_re_min']:.6g} | {summary['v_re_max']:.6g} |",
            f"| `V_im` | {summary['v_im_min']:.6g} | {summary['v_im_max']:.6g} |",
            f"| `n_re` | {summary['n_re_min']:.6g} | {summary['n_re_max']:.6g} |",
            f"| `n_im` | {summary['n_im_min']:.6g} | {summary['n_im_max']:.6g} |",
            "",
            "## Boundary",
            "",
            "- The transfer functions follow the public aIDT MATLAB equations for PTF/ATF, with direct shifted-pupil evaluation rather than integer `circshift`.",
            "- The solver streams normal-equation terms instead of materializing full `PTF_4D` and `ATF_4D` arrays.",
            "- This establishes the physical reconstruction layer needed before comparing a prepared transfer-function operator against package baselines.",
        ]
    )
    if summary.get("figure"):
        lines.extend(["", f"![central reconstruction slices]({summary['figure']})"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, *, n_re: np.ndarray, n_im: np.ndarray, depth_values_um: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_index = int(np.argmin(np.abs(depth_values_um)))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    panels = [
        (axes[0, 0], n_re[:, :, z_index], f"n_re z={depth_values_um[z_index]:.1f} um"),
        (axes[0, 1], n_im[:, :, z_index], f"n_im z={depth_values_um[z_index]:.1f} um"),
        (axes[1, 0], np.max(n_re, axis=2), "n_re max projection"),
        (axes[1, 1], np.max(np.abs(n_im), axis=2), "|n_im| max projection"),
    ]
    for ax, image, title in panels:
        im = ax.imshow(image, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    measured = load_odt_measured_contract(args.contract)
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    if measured.q_layout != "annular_cartesian_stack":
        raise ValueError(f"expected annular_cartesian_stack contract, got {measured.q_layout!r}")

    preprocess_start = time.perf_counter()
    prep = preprocess_aidt_data(
        measured,
        crop_size=None if args.crop_size <= 0 else args.crop_size,
        background_mode=args.background_mode,
        flatfield_mode=args.flatfield_mode,
        frame_scale_mode=args.frame_scale_mode,
    )
    preprocessing_s = time.perf_counter() - preprocess_start
    depth_values_um = make_depth_values(args.depth_min_um, args.depth_max_um, args.depth_step_um)
    complex_dtype = np.complex64 if args.dtype == "complex64" else np.complex128
    source_na_xy = np.asarray(measured["source_na_xy"], dtype=np.float64)
    if args.max_frames is not None:
        source_na_xy = source_na_xy[: int(args.max_frames)]
        data = prep.data[: int(args.max_frames)]
    else:
        data = prep.data
    accumulation = accumulate_transfer_system(
        data=data,
        frequency_x=prep.frequency_x,
        frequency_y=prep.frequency_y,
        source_na_xy=source_na_xy,
        wavelength_um=float(measured["wavelength"]),
        medium_index=float(measured["medium_index"]),
        objective_na=float(measured["objective_na"]),
        depth_values_um=depth_values_um,
        dz_um=float(args.depth_step_um),
        fft_norm=args.fft_norm,
        evanescent_eps=float(args.evanescent_eps),
        complex_dtype=complex_dtype,
    )
    v_re, v_im, n_re, n_im, solve_stats = solve_transfer_system(
        accumulation,
        alpha=float(args.alpha),
        beta=float(args.beta),
        fft_norm=args.fft_norm,
        medium_index=float(measured["medium_index"]),
    )
    summary: dict[str, Any] = {
        "contract": str(args.contract.resolve()),
        "output_npz": str(args.out.resolve()),
        "summary_json": str(args.json_out.resolve()),
        "summary_md": str(args.summary_md.resolve()) if args.summary_md else None,
        "figure": str(args.figure.resolve()) if args.figure else None,
        "validation": report.to_dict(),
        "input_shape": [int(v) for v in np.asarray(measured["data"]).shape],
        "processed_shape": [int(v) for v in data.shape],
        "crop_size": int(prep.crop_size),
        "crop_origin_yx": [int(v) for v in prep.crop_origin_yx],
        "depth_count": int(depth_values_um.size),
        "depth_min_um": float(depth_values_um[0]),
        "depth_max_um": float(depth_values_um[-1]),
        "depth_step_um": float(args.depth_step_um),
        "background_mode": args.background_mode,
        "flatfield_mode": args.flatfield_mode,
        "frame_scale_mode": args.frame_scale_mode,
        "fft_norm": args.fft_norm,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "complex_dtype": args.dtype,
        "wavelength_um": float(measured["wavelength"]),
        "medium_index": float(measured["medium_index"]),
        "objective_na": float(measured["objective_na"]),
        "n_illum": int(data.shape[0]),
        "preprocessing_s": float(preprocessing_s),
        **prep.preprocessing_stats,
        **accumulation.stats,
        **solve_stats,
    }
    summary["total_s"] = float(time.perf_counter() - total_start)
    write_npz(
        args.out,
        v_re=v_re,
        v_im=v_im,
        n_re=n_re,
        n_im=n_im,
        depth_values_um=depth_values_um,
        frequency_x=prep.frequency_x,
        frequency_y=prep.frequency_y,
        summary=summary,
    )
    if args.figure:
        write_figure(args.figure, n_re=n_re, n_im=n_im, depth_values_um=depth_values_um)
    write_json(args.json_out, {"config": vars(args), "summary": summary})
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a public aIDT PTF/ATF transfer-function reconstruction smoke test.")
    p.add_argument("--contract", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--depth-min-um", type=float, default=-25.5)
    p.add_argument("--depth-max-um", type=float, default=25.5)
    p.add_argument("--depth-step-um", type=float, default=1.5)
    p.add_argument("--alpha", type=float, default=1e2)
    p.add_argument("--beta", type=float, default=1e2)
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--fft-norm", choices=["none", "ortho"], default="ortho")
    p.add_argument("--background-mode", choices=["none", "frame_mean", "global_mean", "edge_median"], default="frame_mean")
    p.add_argument("--flatfield-mode", choices=["none", "global_mean"], default="none")
    p.add_argument("--frame-scale-mode", choices=["none", "rms"], default="none")
    p.add_argument("--evanescent-eps", type=float, default=1e-9)
    p.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_reconstruction.npz")
    p.add_argument("--json-out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_reconstruction.json")
    p.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_reconstruction.md")
    p.add_argument("--figure", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_reconstruction.png")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
