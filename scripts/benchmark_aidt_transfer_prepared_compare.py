from __future__ import annotations

import argparse
import gc
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconstruct_aidt_public_transfer_function import (  # noqa: E402
    TransferAccumulation,
    accumulate_transfer_system,
    centered_fft2,
    centered_ifft2,
    json_default,
    load_odt_measured_contract,
    make_depth_values,
    preprocess_aidt_data,
    solve_transfer_system,
    transfer_functions_for_source,
    validate_odt_measured_contract,
)


@dataclass(frozen=True)
class PreparedTransferPlan:
    ptf_stack: np.ndarray
    atf_stack: np.ndarray
    sum_ptf: np.ndarray
    sum_atf: np.ndarray
    conj_ptf_atf: np.ndarray
    conj_atf_ptf: np.ndarray
    source_stats: list[dict[str, Any]]
    setup_s: float
    cache_bytes: int


@dataclass(frozen=True)
class GeometryTransferPlan:
    uv1_stack: np.ndarray
    uv2_stack: np.ndarray
    green1_stack: np.ndarray
    green2_stack: np.ndarray
    pupil1_stack: np.ndarray
    pupil2_stack: np.ndarray
    source_axial: np.ndarray
    sum_ptf: np.ndarray
    sum_atf: np.ndarray
    conj_ptf_atf: np.ndarray
    conj_atf_ptf: np.ndarray
    setup_s: float
    cache_bytes: int


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected.astype(np.float64, copy=False)))
    if denom == 0.0:
        return float(np.linalg.norm(actual.astype(np.float64, copy=False)))
    return float(np.linalg.norm((actual - expected).astype(np.float64, copy=False)) / denom)


def build_prepared_transfer_plan(
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
    complex_dtype: np.dtype,
    image_shape: tuple[int, int],
) -> PreparedTransferPlan:
    h, w = image_shape
    m = int(source_na_xy.shape[0])
    nz = int(depth_values_um.size)
    real_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    ptf_stack = np.empty((m, h, w, nz), dtype=complex_dtype)
    atf_stack = np.empty((m, h, w, nz), dtype=complex_dtype)
    sum_ptf = np.zeros((h, w, nz), dtype=real_dtype)
    sum_atf = np.zeros((h, w, nz), dtype=real_dtype)
    conj_ptf_atf = np.zeros((h, w, nz), dtype=complex_dtype)
    conj_atf_ptf = np.zeros((h, w, nz), dtype=complex_dtype)
    source_stats: list[dict[str, Any]] = []

    start = time.perf_counter()
    for frame_index in range(m):
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
        ptf_stack[frame_index] = ptf
        atf_stack[frame_index] = atf
        sum_ptf += np.real(np.conj(ptf) * ptf)
        sum_atf += np.real(np.conj(atf) * atf)
        conj_ptf_atf += np.conj(ptf) * atf
        conj_atf_ptf += np.conj(atf) * ptf
        source_stats.append({"frame_index": int(frame_index), **stats})
    setup_s = time.perf_counter() - start
    cache_bytes = int(
        ptf_stack.nbytes
        + atf_stack.nbytes
        + sum_ptf.nbytes
        + sum_atf.nbytes
        + conj_ptf_atf.nbytes
        + conj_atf_ptf.nbytes
    )
    return PreparedTransferPlan(
        ptf_stack=ptf_stack,
        atf_stack=atf_stack,
        sum_ptf=sum_ptf,
        sum_atf=sum_atf,
        conj_ptf_atf=conj_ptf_atf,
        conj_atf_ptf=conj_atf_ptf,
        source_stats=source_stats,
        setup_s=float(setup_s),
        cache_bytes=cache_bytes,
    )


def transfer_functions_from_geometry(
    *,
    uv1: np.ndarray,
    uv2: np.ndarray,
    green1: np.ndarray,
    green2: np.ndarray,
    pupil1: np.ndarray,
    pupil2: np.ndarray,
    source_axial: float,
    depth_values_um: np.ndarray,
    dz_um: float,
    wavelength_um: float,
    medium_index: float,
    complex_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    k0 = 2.0 * math.pi / float(wavelength_um)
    k_medium = k0 * float(medium_index)
    z = depth_values_um[None, None, :].astype(np.float32)
    phase1 = k_medium * z * (uv1[:, :, None] - float(source_axial))
    phase2 = k_medium * z * (uv2[:, :, None] - float(source_axial))
    green1_3d = green1[:, :, None]
    green2_3d = green2[:, :, None]
    pupil1_3d = pupil1[:, :, None]
    pupil2_3d = pupil2[:, :, None]
    sin1 = np.sin(phase1)
    sin2 = np.sin(phase2)
    cos1 = np.cos(phase1)
    cos2 = np.cos(phase2)
    term1_sin = pupil1_3d * sin1 * green1_3d
    term2_sin = pupil2_3d * sin2 * green2_3d
    term1_cos = pupil1_3d * cos1 * green1_3d
    term2_cos = pupil2_3d * cos2 * green2_3d
    scale = 0.5 * float(dz_um) * k0 * k0
    ptf = scale * ((term1_sin + term2_sin) + 1j * (term1_cos - term2_cos))
    atf = scale * (-(term1_cos + term2_cos) + 1j * (term1_sin - term2_sin))
    return ptf.astype(complex_dtype, copy=False), atf.astype(complex_dtype, copy=False)


def build_geometry_transfer_plan(
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
    complex_dtype: np.dtype,
    image_shape: tuple[int, int],
) -> GeometryTransferPlan:
    h, w = image_shape
    m = int(source_na_xy.shape[0])
    nz = int(depth_values_um.size)
    real_dtype = np.float32
    fx = frequency_x[None, :].astype(np.float64)
    fy = frequency_y[:, None].astype(np.float64)
    k0 = 2.0 * math.pi / float(wavelength_um)
    k_medium = k0 * float(medium_index)
    max_frequency = float(objective_na) / float(wavelength_um)

    uv1_stack = np.empty((m, h, w), dtype=real_dtype)
    uv2_stack = np.empty((m, h, w), dtype=real_dtype)
    green1_stack = np.empty((m, h, w), dtype=real_dtype)
    green2_stack = np.empty((m, h, w), dtype=real_dtype)
    pupil1_stack = np.empty((m, h, w), dtype=real_dtype)
    pupil2_stack = np.empty((m, h, w), dtype=real_dtype)
    source_axial = np.empty((m,), dtype=real_dtype)
    sum_ptf = np.zeros((h, w, nz), dtype=real_dtype)
    sum_atf = np.zeros((h, w, nz), dtype=real_dtype)
    conj_ptf_atf = np.zeros((h, w, nz), dtype=complex_dtype)
    conj_atf_ptf = np.zeros((h, w, nz), dtype=complex_dtype)

    def axial_and_green(delta_fx: np.ndarray, delta_fy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radial2 = delta_fx * delta_fx + delta_fy * delta_fy
        inside = 1.0 - float(wavelength_um) ** 2 * radial2
        valid = (inside > float(evanescent_eps)) & (radial2 <= max_frequency * max_frequency)
        axial = np.zeros_like(inside, dtype=np.float64)
        axial[valid] = np.sqrt(inside[valid])
        green = np.zeros_like(inside, dtype=np.float64)
        green[valid] = 1.0 / (k_medium * axial[valid])
        return axial.astype(real_dtype), green.astype(real_dtype), valid.astype(real_dtype)

    start = time.perf_counter()
    for frame_index in range(m):
        source_fx = float(source_na_xy[frame_index, 0]) / float(wavelength_um)
        source_fy = float(source_na_xy[frame_index, 1]) / float(wavelength_um)
        uv1, green1, pupil1 = axial_and_green(fx - source_fx, fy - source_fy)
        uv2, green2, pupil2 = axial_and_green(fx + source_fx, fy + source_fy)
        source_inside = 1.0 - float(wavelength_um) ** 2 * (source_fx * source_fx + source_fy * source_fy)
        if source_inside <= float(evanescent_eps):
            raise ValueError(f"source NA is evanescent or grazing: {source_na_xy[frame_index]}")
        uv1_stack[frame_index] = uv1
        uv2_stack[frame_index] = uv2
        green1_stack[frame_index] = green1
        green2_stack[frame_index] = green2
        pupil1_stack[frame_index] = pupil1
        pupil2_stack[frame_index] = pupil2
        source_axial[frame_index] = math.sqrt(source_inside)
        ptf, atf = transfer_functions_from_geometry(
            uv1=uv1,
            uv2=uv2,
            green1=green1,
            green2=green2,
            pupil1=pupil1,
            pupil2=pupil2,
            source_axial=float(source_axial[frame_index]),
            depth_values_um=depth_values_um,
            dz_um=dz_um,
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            complex_dtype=complex_dtype,
        )
        sum_ptf += np.real(np.conj(ptf) * ptf)
        sum_atf += np.real(np.conj(atf) * atf)
        conj_ptf_atf += np.conj(ptf) * atf
        conj_atf_ptf += np.conj(atf) * ptf
    setup_s = time.perf_counter() - start
    cache_bytes = int(
        uv1_stack.nbytes
        + uv2_stack.nbytes
        + green1_stack.nbytes
        + green2_stack.nbytes
        + pupil1_stack.nbytes
        + pupil2_stack.nbytes
        + source_axial.nbytes
        + sum_ptf.nbytes
        + sum_atf.nbytes
        + conj_ptf_atf.nbytes
        + conj_atf_ptf.nbytes
    )
    return GeometryTransferPlan(
        uv1_stack=uv1_stack,
        uv2_stack=uv2_stack,
        green1_stack=green1_stack,
        green2_stack=green2_stack,
        pupil1_stack=pupil1_stack,
        pupil2_stack=pupil2_stack,
        source_axial=source_axial,
        sum_ptf=sum_ptf,
        sum_atf=sum_atf,
        conj_ptf_atf=conj_ptf_atf,
        conj_atf_ptf=conj_atf_ptf,
        setup_s=float(setup_s),
        cache_bytes=cache_bytes,
    )


def geometry_reconstruct(
    *,
    plan: GeometryTransferPlan,
    data: np.ndarray,
    depth_values_um: np.ndarray,
    dz_um: float,
    wavelength_um: float,
    medium_index: float,
    alpha: float,
    beta: float,
    fft_norm: str,
    complex_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = data.shape[1:]
    nz = int(plan.sum_ptf.shape[2])
    conj_ptf_image = np.zeros((h, w, nz), dtype=complex_dtype)
    conj_atf_image = np.zeros((h, w, nz), dtype=complex_dtype)
    fft_norms: list[float] = []

    rhs_start = time.perf_counter()
    for frame_index in range(data.shape[0]):
        image_fft = centered_fft2(data[frame_index], norm=fft_norm).astype(complex_dtype, copy=False)
        fft_norms.append(float(np.linalg.norm(image_fft.astype(np.complex128))))
        ptf, atf = transfer_functions_from_geometry(
            uv1=plan.uv1_stack[frame_index],
            uv2=plan.uv2_stack[frame_index],
            green1=plan.green1_stack[frame_index],
            green2=plan.green2_stack[frame_index],
            pupil1=plan.pupil1_stack[frame_index],
            pupil2=plan.pupil2_stack[frame_index],
            source_axial=float(plan.source_axial[frame_index]),
            depth_values_um=depth_values_um,
            dz_um=dz_um,
            wavelength_um=wavelength_um,
            medium_index=medium_index,
            complex_dtype=complex_dtype,
        )
        conj_ptf_image += np.conj(ptf) * image_fft[:, :, None]
        conj_atf_image += np.conj(atf) * image_fft[:, :, None]
    rhs_s = time.perf_counter() - rhs_start
    accumulation = TransferAccumulation(
        sum_ptf=plan.sum_ptf,
        sum_atf=plan.sum_atf,
        conj_ptf_image=conj_ptf_image,
        conj_atf_image=conj_atf_image,
        conj_ptf_atf=plan.conj_ptf_atf,
        conj_atf_ptf=plan.conj_atf_ptf,
        stats={
            "accumulation_s": float(rhs_s),
            "fft_abs_norm_min": float(np.min(fft_norms)),
            "fft_abs_norm_max": float(np.max(fft_norms)),
            "sum_ptf_max": float(np.max(plan.sum_ptf)),
            "sum_atf_max": float(np.max(plan.sum_atf)),
            "sum_ptf_nonzero_fraction": float(np.mean(plan.sum_ptf > 0)),
            "sum_atf_nonzero_fraction": float(np.mean(plan.sum_atf > 0)),
        },
    )
    v_re, v_im, n_re, n_im, solve_stats = solve_transfer_system(
        accumulation,
        alpha=alpha,
        beta=beta,
        fft_norm=fft_norm,
        medium_index=medium_index,
    )
    stats = {**accumulation.stats, **solve_stats, "rhs_s": float(rhs_s)}
    stats["run_s"] = float(rhs_s + solve_stats["solve_s"])
    return v_re, v_im, n_re, n_im, stats


def solve_block_to_frequency(
    *,
    sum_ptf: np.ndarray,
    sum_atf: np.ndarray,
    conj_ptf_image: np.ndarray,
    conj_atf_image: np.ndarray,
    conj_ptf_atf: np.ndarray,
    conj_atf_ptf: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    denom = (sum_ptf + float(alpha)) * (sum_atf + float(beta)) - (conj_ptf_atf * conj_atf_ptf)
    denom_abs = np.abs(denom)
    denom_floor = max(float(np.max(denom_abs)) * 1e-12, 1e-30)
    denom = np.where(denom_abs > denom_floor, denom, denom_floor + 0j)
    v_re_freq = ((sum_atf + float(beta)) * conj_ptf_image - conj_ptf_atf * conj_atf_image) / denom
    v_im_freq = ((sum_ptf + float(alpha)) * conj_atf_image - conj_atf_ptf * conj_ptf_image) / denom
    return v_re_freq, v_im_freq, {
        "denom_abs_min": float(np.min(denom_abs)),
        "denom_abs_max": float(np.max(denom_abs)),
        "denom_floor": float(denom_floor),
    }


def ri_from_potential(
    *,
    v_re: np.ndarray,
    v_im: np.ndarray,
    medium_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    n2_plus_v = float(medium_index) ** 2 + v_re
    n_re = np.sqrt(np.maximum((n2_plus_v + np.sqrt(np.maximum(n2_plus_v * n2_plus_v + v_im * v_im, 0.0))) / 2.0, 0.0))
    n_im = np.divide(v_im, 2.0 * np.maximum(n_re, 1e-12))
    return n_re.astype(np.float32), n_im.astype(np.float32)


def blocked_geometry_reconstruct(
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
    evanescent_eps: float,
    alpha: float,
    beta: float,
    fft_norm: str,
    complex_dtype: np.dtype,
    block_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = data.shape[1:]
    nz = int(depth_values_um.size)
    real_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    block_rows = max(1, min(int(block_rows), h))
    n_blocks = int(math.ceil(h / block_rows))

    fft_start = time.perf_counter()
    fft_stack = np.empty((data.shape[0], h, w), dtype=complex_dtype)
    fft_norms: list[float] = []
    for frame_index in range(data.shape[0]):
        image_fft = centered_fft2(data[frame_index], norm=fft_norm).astype(complex_dtype, copy=False)
        fft_stack[frame_index] = image_fft
        fft_norms.append(float(np.linalg.norm(image_fft.astype(np.complex128))))
    fft_s = time.perf_counter() - fft_start

    v_re_freq = np.empty((h, w, nz), dtype=complex_dtype)
    v_im_freq = np.empty((h, w, nz), dtype=complex_dtype)
    accumulation_s = 0.0
    block_solve_s = 0.0
    denom_min = float("inf")
    denom_max = 0.0
    denom_floor_max = 0.0
    sum_ptf_max = 0.0
    sum_atf_max = 0.0
    nonzero_ptf = 0
    nonzero_atf = 0
    total_entries = h * w * nz
    peak_block_bytes = 0

    for y0 in range(0, h, block_rows):
        y1 = min(h, y0 + block_rows)
        bh = y1 - y0
        block_shape = (bh, w, nz)
        sum_ptf = np.zeros(block_shape, dtype=real_dtype)
        sum_atf = np.zeros(block_shape, dtype=real_dtype)
        conj_ptf_image = np.zeros(block_shape, dtype=complex_dtype)
        conj_atf_image = np.zeros(block_shape, dtype=complex_dtype)
        conj_ptf_atf = np.zeros(block_shape, dtype=complex_dtype)
        conj_atf_ptf = np.zeros(block_shape, dtype=complex_dtype)
        peak_block_bytes = max(
            peak_block_bytes,
            sum_ptf.nbytes
            + sum_atf.nbytes
            + conj_ptf_image.nbytes
            + conj_atf_image.nbytes
            + conj_ptf_atf.nbytes
            + conj_atf_ptf.nbytes,
        )
        acc_start = time.perf_counter()
        for frame_index in range(data.shape[0]):
            ptf, atf, _ = transfer_functions_for_source(
                frequency_x=frequency_x,
                frequency_y=frequency_y[y0:y1],
                source_na_xy=source_na_xy[frame_index],
                wavelength_um=wavelength_um,
                medium_index=medium_index,
                objective_na=objective_na,
                depth_values_um=depth_values_um,
                dz_um=dz_um,
                evanescent_eps=evanescent_eps,
                dtype=complex_dtype,
            )
            image_fft = fft_stack[frame_index, y0:y1, :]
            sum_ptf += np.real(np.conj(ptf) * ptf)
            sum_atf += np.real(np.conj(atf) * atf)
            conj_ptf_image += np.conj(ptf) * image_fft[:, :, None]
            conj_atf_image += np.conj(atf) * image_fft[:, :, None]
            conj_ptf_atf += np.conj(ptf) * atf
            conj_atf_ptf += np.conj(atf) * ptf
        accumulation_s += time.perf_counter() - acc_start

        solve_start = time.perf_counter()
        v_re_block, v_im_block, block_stats = solve_block_to_frequency(
            sum_ptf=sum_ptf,
            sum_atf=sum_atf,
            conj_ptf_image=conj_ptf_image,
            conj_atf_image=conj_atf_image,
            conj_ptf_atf=conj_ptf_atf,
            conj_atf_ptf=conj_atf_ptf,
            alpha=alpha,
            beta=beta,
        )
        v_re_freq[y0:y1] = v_re_block
        v_im_freq[y0:y1] = v_im_block
        block_solve_s += time.perf_counter() - solve_start
        denom_min = min(denom_min, float(block_stats["denom_abs_min"]))
        denom_max = max(denom_max, float(block_stats["denom_abs_max"]))
        denom_floor_max = max(denom_floor_max, float(block_stats["denom_floor"]))
        sum_ptf_max = max(sum_ptf_max, float(np.max(sum_ptf)))
        sum_atf_max = max(sum_atf_max, float(np.max(sum_atf)))
        nonzero_ptf += int(np.count_nonzero(sum_ptf > 0))
        nonzero_atf += int(np.count_nonzero(sum_atf > 0))

    ifft_start = time.perf_counter()
    v_re = np.real(centered_ifft2(v_re_freq, norm=fft_norm, axes=(0, 1))).astype(np.float32)
    v_im = np.real(centered_ifft2(v_im_freq, norm=fft_norm, axes=(0, 1))).astype(np.float32)
    n_re, n_im = ri_from_potential(v_re=v_re, v_im=v_im, medium_index=medium_index)
    ifft_s = time.perf_counter() - ifft_start
    working_bytes = int(
        fft_stack.nbytes
        + v_re_freq.nbytes
        + v_im_freq.nbytes
        + peak_block_bytes
    )
    stats = {
        "run_s": float(fft_s + accumulation_s + block_solve_s + ifft_s),
        "fft_s": float(fft_s),
        "accumulation_s": float(accumulation_s),
        "block_solve_s": float(block_solve_s),
        "ifft_s": float(ifft_s),
        "solve_s": float(block_solve_s + ifft_s),
        "rhs_s": float(fft_s + accumulation_s),
        "block_rows": int(block_rows),
        "n_blocks": int(n_blocks),
        "working_mib_estimate": float(working_bytes / (1024.0 * 1024.0)),
        "persistent_cache_mib": 0.0,
        "fft_abs_norm_min": float(np.min(fft_norms)),
        "fft_abs_norm_max": float(np.max(fft_norms)),
        "sum_ptf_max": float(sum_ptf_max),
        "sum_atf_max": float(sum_atf_max),
        "sum_ptf_nonzero_fraction": float(nonzero_ptf / max(total_entries, 1)),
        "sum_atf_nonzero_fraction": float(nonzero_atf / max(total_entries, 1)),
        "denom_abs_min": float(denom_min),
        "denom_abs_max": float(denom_max),
        "denom_floor": float(denom_floor_max),
        "v_re_min": float(np.min(v_re)),
        "v_re_max": float(np.max(v_re)),
        "v_im_min": float(np.min(v_im)),
        "v_im_max": float(np.max(v_im)),
        "n_re_min": float(np.min(n_re)),
        "n_re_max": float(np.max(n_re)),
        "n_im_min": float(np.min(n_im)),
        "n_im_max": float(np.max(n_im)),
    }
    return v_re, v_im, n_re, n_im, stats


def prepared_reconstruct(
    *,
    plan: PreparedTransferPlan,
    data: np.ndarray,
    alpha: float,
    beta: float,
    fft_norm: str,
    medium_index: float,
    complex_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = data.shape[1:]
    nz = int(plan.sum_ptf.shape[2])
    conj_ptf_image = np.zeros((h, w, nz), dtype=complex_dtype)
    conj_atf_image = np.zeros((h, w, nz), dtype=complex_dtype)
    fft_norms: list[float] = []

    rhs_start = time.perf_counter()
    for frame_index in range(data.shape[0]):
        image_fft = centered_fft2(data[frame_index], norm=fft_norm).astype(complex_dtype, copy=False)
        fft_norms.append(float(np.linalg.norm(image_fft.astype(np.complex128))))
        conj_ptf_image += np.conj(plan.ptf_stack[frame_index]) * image_fft[:, :, None]
        conj_atf_image += np.conj(plan.atf_stack[frame_index]) * image_fft[:, :, None]
    rhs_s = time.perf_counter() - rhs_start
    accumulation = TransferAccumulation(
        sum_ptf=plan.sum_ptf,
        sum_atf=plan.sum_atf,
        conj_ptf_image=conj_ptf_image,
        conj_atf_image=conj_atf_image,
        conj_ptf_atf=plan.conj_ptf_atf,
        conj_atf_ptf=plan.conj_atf_ptf,
        stats={
            "accumulation_s": float(rhs_s),
            "fft_abs_norm_min": float(np.min(fft_norms)),
            "fft_abs_norm_max": float(np.max(fft_norms)),
            "sum_ptf_max": float(np.max(plan.sum_ptf)),
            "sum_atf_max": float(np.max(plan.sum_atf)),
            "sum_ptf_nonzero_fraction": float(np.mean(plan.sum_ptf > 0)),
            "sum_atf_nonzero_fraction": float(np.mean(plan.sum_atf > 0)),
        },
    )
    v_re, v_im, n_re, n_im, solve_stats = solve_transfer_system(
        accumulation,
        alpha=alpha,
        beta=beta,
        fft_norm=fft_norm,
        medium_index=medium_index,
    )
    stats = {**accumulation.stats, **solve_stats, "rhs_s": float(rhs_s)}
    stats["run_s"] = float(rhs_s + solve_stats["solve_s"])
    return v_re, v_im, n_re, n_im, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Public aIDT prepared-transfer comparison",
        "",
        "This benchmark compares the same public aIDT PTF/ATF reconstruction equation in three execution modes.",
        "",
        "- `streaming`: regenerate transfer functions and accumulate the normal-equation system for every reconstruction.",
        "- `prepared_cache`: build and store the full PTF/ATF transfer stack plus geometry-only normal-equation terms once, then reuse them for repeated reconstructions.",
        "- `geometry_cache`: cache only 2D geometry maps and 3D normal-equation terms; regenerate z-dependent PTF/ATF values for the RHS.",
        "",
        "## Repeated-Run Results",
        "",
        "| crop | method | setup s | run median s | run speedup | break-even repeats | cache MiB | working MiB | n_re rel-L2 | n_im rel-L2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        for method_key, label in (
            ("prepared", "prepared_cache"),
            ("geometry", "geometry_cache"),
            ("blocked", "blocked_geometry_cache"),
        ):
            if f"{method_key}_run_median_s" not in case:
                continue
            lines.append(
                "| {crop} | `{label}` | {setup:.6f} | {run:.6f} | {speed:.3f} | {break_even:.2f} | {cache:.1f} | {working:.1f} | {nre:.3g} | {nim:.3g} |".format(
                    crop=case["crop_size"],
                    label=label,
                    setup=case[f"{method_key}_setup_s"],
                    run=case[f"{method_key}_run_median_s"],
                    speed=case[f"{method_key}_run_speedup_vs_streaming"],
                    break_even=case[f"{method_key}_break_even_repeats"],
                    cache=case[f"{method_key}_cache_mib"],
                    working=case[f"{method_key}_working_mib_estimate"],
                    nre=case[f"{method_key}_n_re_rel_l2_vs_streaming"],
                    nim=case[f"{method_key}_n_im_rel_l2_vs_streaming"],
                )
            )
    lines.extend(
        [
            "",
            "For one-shot use, the prepared total is `setup + run`, so repeated geometry reuse is the main advantageous regime.",
            "",
            "| crop | method | one-shot s | one-shot ratio vs streaming |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        for method_key, label in (
            ("prepared", "prepared_cache"),
            ("geometry", "geometry_cache"),
            ("blocked", "blocked_geometry_cache"),
        ):
            if f"{method_key}_one_shot_total_s" not in case:
                continue
            lines.append(
                f"| {case['crop_size']} | `{label}` | {case[f'{method_key}_one_shot_total_s']:.6f} | {case[f'{method_key}_one_shot_ratio_vs_streaming']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a same-equation comparison, so numerical differences should stay near floating-point accumulation error.",
            "- `prepared_cache` is the fastest repeated-run path but stores the full 4D transfer stack.",
            "- `geometry_cache` is the memory-reduced path; it trades more per-run trigonometric work for a much smaller cache.",
            "- `blocked_geometry_cache` is the low-memory path; it keeps only output spectra plus one frequency-row block of working arrays.",
            "- The current implementation is NumPy CPU code. It has not yet used GPU kernels or low-level C++ fused loops for the transfer-function RHS.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(
    *,
    args: argparse.Namespace,
    measured: Any,
    crop_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complex_dtype = np.complex64 if args.dtype == "complex64" else np.complex128
    prep = preprocess_aidt_data(
        measured,
        crop_size=crop_size,
        background_mode=args.background_mode,
        flatfield_mode=args.flatfield_mode,
        frame_scale_mode=args.frame_scale_mode,
    )
    source_na_xy = np.asarray(measured["source_na_xy"], dtype=np.float64)
    data = prep.data
    if args.max_frames is not None:
        data = data[: int(args.max_frames)]
        source_na_xy = source_na_xy[: int(args.max_frames)]
    depth_values_um = make_depth_values(args.depth_min_um, args.depth_max_um, args.depth_step_um)
    common = {
        "frequency_x": prep.frequency_x,
        "frequency_y": prep.frequency_y,
        "source_na_xy": source_na_xy,
        "wavelength_um": float(measured["wavelength"]),
        "medium_index": float(measured["medium_index"]),
        "objective_na": float(measured["objective_na"]),
        "depth_values_um": depth_values_um,
        "dz_um": float(args.depth_step_um),
        "evanescent_eps": float(args.evanescent_eps),
        "complex_dtype": complex_dtype,
    }

    streaming_rows: list[dict[str, Any]] = []
    streaming_outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    streaming_total_times: list[float] = []
    for repeat in range(max(1, int(args.streaming_repeats))):
        start = time.perf_counter()
        accumulation = accumulate_transfer_system(
            data=data,
            fft_norm=args.fft_norm,
            **common,
        )
        v_re, v_im, n_re, n_im, solve_stats = solve_transfer_system(
            accumulation,
            alpha=float(args.alpha),
            beta=float(args.beta),
            fft_norm=args.fft_norm,
            medium_index=float(measured["medium_index"]),
        )
        total_s = time.perf_counter() - start
        streaming_total_times.append(float(total_s))
        if streaming_outputs is None:
            streaming_outputs = (v_re, v_im, n_re, n_im)
        streaming_rows.append(
            {
                "crop_size": int(crop_size),
                "method": "streaming",
                "repeat": int(repeat),
                "setup_s": 0.0,
                "run_s": float(total_s),
                "accumulation_s": float(accumulation.stats["accumulation_s"]),
                "solve_s": float(solve_stats["solve_s"]),
                "cache_mib": 0.0,
            }
        )

    if streaming_outputs is None:
        raise RuntimeError("streaming reference did not run")

    streaming_reference = streaming_outputs
    streaming_total = float(median(streaming_total_times))
    case_summary = {
        "crop_size": int(crop_size),
        "processed_shape": [int(v) for v in data.shape],
        "depth_count": int(depth_values_um.size),
        "n_illum": int(data.shape[0]),
        "streaming_total_s": streaming_total,
        "streaming_accumulation_median_s": float(median(row["accumulation_s"] for row in streaming_rows)),
        "streaming_solve_median_s": float(median(row["solve_s"] for row in streaming_rows)),
    }

    prepared_rows: list[dict[str, Any]] = []
    if not args.skip_prepared_cache:
        plan = build_prepared_transfer_plan(
            image_shape=tuple(int(v) for v in data.shape[1:]),
            **common,
        )
        prepared_run_times: list[float] = []
        prepared_rhs_times: list[float] = []
        prepared_solve_times: list[float] = []
        last_prepared_outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for repeat in range(max(1, int(args.prepared_repeats))):
            v_re, v_im, n_re, n_im, stats = prepared_reconstruct(
                plan=plan,
                data=data,
                alpha=float(args.alpha),
                beta=float(args.beta),
                fft_norm=args.fft_norm,
                medium_index=float(measured["medium_index"]),
                complex_dtype=complex_dtype,
            )
            last_prepared_outputs = (v_re, v_im, n_re, n_im)
            prepared_run_times.append(float(stats["run_s"]))
            prepared_rhs_times.append(float(stats["rhs_s"]))
            prepared_solve_times.append(float(stats["solve_s"]))
            prepared_rows.append(
                {
                    "crop_size": int(crop_size),
                    "method": "prepared_cache",
                    "repeat": int(repeat),
                    "setup_s": float(plan.setup_s) if repeat == 0 else 0.0,
                    "run_s": float(stats["run_s"]),
                    "accumulation_s": float(stats["rhs_s"]),
                    "solve_s": float(stats["solve_s"]),
                    "cache_mib": float(plan.cache_bytes / (1024.0 * 1024.0)),
                    "working_mib_estimate": float(plan.cache_bytes / (1024.0 * 1024.0)),
                }
            )

        if last_prepared_outputs is None:
            raise RuntimeError("prepared reconstruction did not run")
        prepared_reference = last_prepared_outputs
        prepared_run = float(median(prepared_run_times))
        speedup = float(streaming_total / prepared_run) if prepared_run > 0 else float("inf")
        denom = streaming_total - prepared_run
        break_even = float(plan.setup_s / denom + 1.0) if denom > 0 else float("inf")
        prepared_setup_s = float(plan.setup_s)
        prepared_cache_mib = float(plan.cache_bytes / (1024.0 * 1024.0))
        case_summary.update(
            {
                "prepared_setup_s": prepared_setup_s,
                "prepared_one_shot_total_s": float(prepared_setup_s + prepared_run),
                "prepared_one_shot_ratio_vs_streaming": float((prepared_setup_s + prepared_run) / streaming_total),
                "prepared_run_median_s": prepared_run,
                "prepared_rhs_median_s": float(median(prepared_rhs_times)),
                "prepared_solve_median_s": float(median(prepared_solve_times)),
                "prepared_run_speedup_vs_streaming": speedup,
                "prepared_break_even_repeats": break_even,
                "prepared_cache_mib": prepared_cache_mib,
                "prepared_working_mib_estimate": prepared_cache_mib,
                "prepared_n_re_rel_l2_vs_streaming": rel_l2(prepared_reference[2], streaming_reference[2]),
                "prepared_n_im_rel_l2_vs_streaming": rel_l2(prepared_reference[3], streaming_reference[3]),
                "prepared_v_re_rel_l2_vs_streaming": rel_l2(prepared_reference[0], streaming_reference[0]),
                "prepared_v_im_rel_l2_vs_streaming": rel_l2(prepared_reference[1], streaming_reference[1]),
            }
        )
        del plan
        gc.collect()

    geometry_rows: list[dict[str, Any]] = []
    if not args.skip_geometry_cache:
        geometry_plan = build_geometry_transfer_plan(
            image_shape=tuple(int(v) for v in data.shape[1:]),
            **common,
        )
        geometry_run_times: list[float] = []
        geometry_rhs_times: list[float] = []
        geometry_solve_times: list[float] = []
        last_geometry_outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for repeat in range(max(1, int(args.prepared_repeats))):
            v_re, v_im, n_re, n_im, stats = geometry_reconstruct(
                plan=geometry_plan,
                data=data,
                depth_values_um=depth_values_um,
                dz_um=float(args.depth_step_um),
                wavelength_um=float(measured["wavelength"]),
                medium_index=float(measured["medium_index"]),
                alpha=float(args.alpha),
                beta=float(args.beta),
                fft_norm=args.fft_norm,
                complex_dtype=complex_dtype,
            )
            last_geometry_outputs = (v_re, v_im, n_re, n_im)
            geometry_run_times.append(float(stats["run_s"]))
            geometry_rhs_times.append(float(stats["rhs_s"]))
            geometry_solve_times.append(float(stats["solve_s"]))
            geometry_rows.append(
                {
                    "crop_size": int(crop_size),
                    "method": "geometry_cache",
                    "repeat": int(repeat),
                    "setup_s": float(geometry_plan.setup_s) if repeat == 0 else 0.0,
                    "run_s": float(stats["run_s"]),
                    "accumulation_s": float(stats["rhs_s"]),
                    "solve_s": float(stats["solve_s"]),
                    "cache_mib": float(geometry_plan.cache_bytes / (1024.0 * 1024.0)),
                    "working_mib_estimate": float(geometry_plan.cache_bytes / (1024.0 * 1024.0)),
                }
            )
        if last_geometry_outputs is None:
            raise RuntimeError("geometry-cache reconstruction did not run")
        geometry_run = float(median(geometry_run_times))
        geometry_setup_s = float(geometry_plan.setup_s)
        geometry_denom = streaming_total - geometry_run
        geometry_break_even = float(geometry_setup_s / geometry_denom + 1.0) if geometry_denom > 0 else float("inf")
        case_summary.update(
            {
                "geometry_setup_s": geometry_setup_s,
                "geometry_one_shot_total_s": float(geometry_setup_s + geometry_run),
                "geometry_one_shot_ratio_vs_streaming": float((geometry_setup_s + geometry_run) / streaming_total),
                "geometry_run_median_s": geometry_run,
                "geometry_rhs_median_s": float(median(geometry_rhs_times)),
                "geometry_solve_median_s": float(median(geometry_solve_times)),
                "geometry_run_speedup_vs_streaming": float(streaming_total / geometry_run) if geometry_run > 0 else float("inf"),
                "geometry_break_even_repeats": geometry_break_even,
                "geometry_cache_mib": float(geometry_plan.cache_bytes / (1024.0 * 1024.0)),
                "geometry_working_mib_estimate": float(geometry_plan.cache_bytes / (1024.0 * 1024.0)),
                "geometry_n_re_rel_l2_vs_streaming": rel_l2(last_geometry_outputs[2], streaming_reference[2]),
                "geometry_n_im_rel_l2_vs_streaming": rel_l2(last_geometry_outputs[3], streaming_reference[3]),
                "geometry_v_re_rel_l2_vs_streaming": rel_l2(last_geometry_outputs[0], streaming_reference[0]),
                "geometry_v_im_rel_l2_vs_streaming": rel_l2(last_geometry_outputs[1], streaming_reference[1]),
            }
        )
        del geometry_plan
        gc.collect()

    blocked_rows: list[dict[str, Any]] = []
    if not args.skip_blocked_geometry_cache:
        blocked_run_times: list[float] = []
        blocked_rhs_times: list[float] = []
        blocked_solve_times: list[float] = []
        blocked_working_mib: list[float] = []
        last_blocked_outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for repeat in range(max(1, int(args.prepared_repeats))):
            v_re, v_im, n_re, n_im, stats = blocked_geometry_reconstruct(
                data=data,
                frequency_x=prep.frequency_x,
                frequency_y=prep.frequency_y,
                source_na_xy=source_na_xy,
                wavelength_um=float(measured["wavelength"]),
                medium_index=float(measured["medium_index"]),
                objective_na=float(measured["objective_na"]),
                depth_values_um=depth_values_um,
                dz_um=float(args.depth_step_um),
                evanescent_eps=float(args.evanescent_eps),
                alpha=float(args.alpha),
                beta=float(args.beta),
                fft_norm=args.fft_norm,
                complex_dtype=complex_dtype,
                block_rows=int(args.block_rows),
            )
            last_blocked_outputs = (v_re, v_im, n_re, n_im)
            blocked_run_times.append(float(stats["run_s"]))
            blocked_rhs_times.append(float(stats["rhs_s"]))
            blocked_solve_times.append(float(stats["solve_s"]))
            blocked_working_mib.append(float(stats["working_mib_estimate"]))
            blocked_rows.append(
                {
                    "crop_size": int(crop_size),
                    "method": "blocked_geometry_cache",
                    "repeat": int(repeat),
                    "setup_s": 0.0,
                    "run_s": float(stats["run_s"]),
                    "accumulation_s": float(stats["accumulation_s"]),
                    "solve_s": float(stats["solve_s"]),
                    "cache_mib": 0.0,
                    "working_mib_estimate": float(stats["working_mib_estimate"]),
                    "block_rows": int(stats["block_rows"]),
                    "n_blocks": int(stats["n_blocks"]),
                    "fft_s": float(stats["fft_s"]),
                    "block_solve_s": float(stats["block_solve_s"]),
                    "ifft_s": float(stats["ifft_s"]),
                }
            )
        if last_blocked_outputs is None:
            raise RuntimeError("blocked geometry reconstruction did not run")
        blocked_run = float(median(blocked_run_times))
        blocked_denom = streaming_total - blocked_run
        blocked_break_even = 1.0 if blocked_denom > 0 else float("inf")
        case_summary.update(
            {
                "blocked_setup_s": 0.0,
                "blocked_one_shot_total_s": blocked_run,
                "blocked_one_shot_ratio_vs_streaming": float(blocked_run / streaming_total),
                "blocked_run_median_s": blocked_run,
                "blocked_rhs_median_s": float(median(blocked_rhs_times)),
                "blocked_solve_median_s": float(median(blocked_solve_times)),
                "blocked_run_speedup_vs_streaming": float(streaming_total / blocked_run) if blocked_run > 0 else float("inf"),
                "blocked_break_even_repeats": blocked_break_even,
                "blocked_cache_mib": 0.0,
                "blocked_working_mib_estimate": float(median(blocked_working_mib)),
                "blocked_block_rows": int(args.block_rows),
                "blocked_n_re_rel_l2_vs_streaming": rel_l2(last_blocked_outputs[2], streaming_reference[2]),
                "blocked_n_im_rel_l2_vs_streaming": rel_l2(last_blocked_outputs[3], streaming_reference[3]),
                "blocked_v_re_rel_l2_vs_streaming": rel_l2(last_blocked_outputs[0], streaming_reference[0]),
                "blocked_v_im_rel_l2_vs_streaming": rel_l2(last_blocked_outputs[1], streaming_reference[1]),
            }
        )
        gc.collect()

    return case_summary, streaming_rows + prepared_rows + geometry_rows + blocked_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    measured = load_odt_measured_contract(args.contract)
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    if measured.q_layout != "annular_cartesian_stack":
        raise ValueError(f"expected annular_cartesian_stack contract, got {measured.q_layout!r}")

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for crop_size in args.crop_sizes:
        case, case_rows = run_case(args=args, measured=measured, crop_size=int(crop_size))
        cases.append(case)
        rows.extend(case_rows)
    summary = {
        "contract": str(args.contract.resolve()),
        "dtype": args.dtype,
        "fft_norm": args.fft_norm,
        "background_mode": args.background_mode,
        "flatfield_mode": args.flatfield_mode,
        "frame_scale_mode": args.frame_scale_mode,
        "depth_min_um": float(args.depth_min_um),
        "depth_max_um": float(args.depth_max_um),
        "depth_step_um": float(args.depth_step_um),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "streaming_repeats": int(args.streaming_repeats),
        "prepared_repeats": int(args.prepared_repeats),
        "skip_prepared_cache": bool(args.skip_prepared_cache),
        "skip_geometry_cache": bool(args.skip_geometry_cache),
        "skip_blocked_geometry_cache": bool(args.skip_blocked_geometry_cache),
        "block_rows": int(args.block_rows),
        "total_s": float(time.perf_counter() - total_start),
        "validation": report.to_dict(),
        "cases": cases,
        "history_csv": str(args.csv.resolve()),
        "summary_md": str(args.summary_md.resolve()) if args.summary_md else None,
    }
    payload = {"config": vars(args), "summary": summary, "rows": rows}
    write_csv(args.csv, rows)
    write_json(args.out, payload)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare public aIDT transfer reconstruction with and without prepared transfer caches.")
    p.add_argument("--contract", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    p.add_argument("--crop-sizes", type=int, nargs="+", default=[64, 128, 256])
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
    p.add_argument("--streaming-repeats", type=int, default=1)
    p.add_argument("--prepared-repeats", type=int, default=3)
    p.add_argument("--skip-prepared-cache", action="store_true")
    p.add_argument("--skip-geometry-cache", action="store_true")
    p.add_argument("--skip-blocked-geometry-cache", action="store_true")
    p.add_argument("--block-rows", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_prepared_compare.json")
    p.add_argument("--csv", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_prepared_compare_history.csv")
    p.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_prepared_compare.md")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
