from __future__ import annotations

import argparse
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

from benchmark_aidt_transfer_prepared_compare import rel_l2  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from reconstruct_aidt_public_transfer_function import (  # noqa: E402
    accumulate_transfer_system,
    json_default,
    load_odt_measured_contract,
    make_depth_values,
    preprocess_aidt_data,
    solve_transfer_system,
    validate_odt_measured_contract,
)


@dataclass(frozen=True)
class TorchAidtGeometryPlan:
    torch: Any
    device: Any
    real_dtype: Any
    complex_dtype: Any
    uv1: Any
    uv2: Any
    green1: Any
    green2: Any
    pupil1: Any
    pupil2: Any
    source_axial: Any
    sum_ptf: Any
    sum_atf: Any
    conj_ptf_atf: Any
    conj_atf_ptf: Any
    rhs_mode: str
    support_indices: Any
    support_uv1: Any
    support_uv2: Any
    support_green1: Any
    support_green2: Any
    support_pupil1: Any
    support_pupil2: Any
    support_conj_ptf: Any
    support_conj_atf: Any
    support_active_fraction: float
    cache_support_transfer: bool
    cache_solve_coeffs: bool
    solve_coeff_v_re_ptf: Any
    solve_coeff_v_re_atf: Any
    solve_coeff_v_im_ptf: Any
    solve_coeff_v_im_atf: Any
    solve_denom_abs_min: float
    solve_denom_abs_max: float
    solve_denom_floor: float
    depth_values: Any
    wavelength_um: float
    medium_index: float
    dz_um: float
    setup_s: float
    cache_bytes: int


def torch_dtypes(torch: Any, dtype: str) -> tuple[Any, Any, Any, Any]:
    if dtype == "complex64":
        return torch.complex64, torch.float32, np.complex64, np.float32
    if dtype == "complex128":
        return torch.complex128, torch.float64, np.complex128, np.float64
    raise ValueError("dtype must be complex64 or complex128")


def tensor_bytes(*values: Any) -> int:
    total = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            total += tensor_bytes(*value)
            continue
        total += int(value.numel() * value.element_size())
    return total


def centered_fft2_torch(torch: Any, image: Any, *, norm: str) -> Any:
    fft_norm = None if norm == "none" else norm
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1), norm=fft_norm),
        dim=(-2, -1),
    )


def centered_ifft2_torch(torch: Any, spectrum: Any, *, norm: str) -> Any:
    fft_norm = None if norm == "none" else norm
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(0, 1)), dim=(0, 1), norm=fft_norm),
        dim=(0, 1),
    )


def transfer_from_geometry_torch(
    *,
    torch: Any,
    uv1: Any,
    uv2: Any,
    green1: Any,
    green2: Any,
    pupil1: Any,
    pupil2: Any,
    source_axial: Any,
    depth_values: Any,
    dz_um: float,
    wavelength_um: float,
    medium_index: float,
    complex_dtype: Any,
) -> tuple[Any, Any]:
    k0 = 2.0 * math.pi / float(wavelength_um)
    k_medium = k0 * float(medium_index)
    z = depth_values.reshape(*([1] * int(uv1.ndim)), -1)
    phase1 = k_medium * z * (uv1[..., None] - source_axial)
    phase2 = k_medium * z * (uv2[..., None] - source_axial)
    green1_3d = green1[..., None]
    green2_3d = green2[..., None]
    pupil1_3d = pupil1[..., None]
    pupil2_3d = pupil2[..., None]
    sin1 = torch.sin(phase1)
    sin2 = torch.sin(phase2)
    cos1 = torch.cos(phase1)
    cos2 = torch.cos(phase2)
    term1_sin = pupil1_3d * sin1 * green1_3d
    term2_sin = pupil2_3d * sin2 * green2_3d
    term1_cos = pupil1_3d * cos1 * green1_3d
    term2_cos = pupil2_3d * cos2 * green2_3d
    scale = 0.5 * float(dz_um) * k0 * k0
    ptf = torch.complex(scale * (term1_sin + term2_sin), scale * (term1_cos - term2_cos))
    atf = torch.complex(-scale * (term1_cos + term2_cos), scale * (term1_sin - term2_sin))
    return ptf.to(dtype=complex_dtype), atf.to(dtype=complex_dtype)


def build_torch_geometry_plan(
    *,
    torch: Any,
    device: Any,
    frequency_x: np.ndarray,
    frequency_y: np.ndarray,
    source_na_xy: np.ndarray,
    wavelength_um: float,
    medium_index: float,
    objective_na: float,
    depth_values_um: np.ndarray,
    dz_um: float,
    evanescent_eps: float,
    dtype: str,
    rhs_mode: str,
    alpha: float,
    beta: float,
    cache_support_transfer: bool,
    cache_solve_coeffs: bool,
) -> TorchAidtGeometryPlan:
    complex_dtype, real_dtype, _, _ = torch_dtypes(torch, dtype)
    h = int(frequency_y.size)
    w = int(frequency_x.size)
    m = int(source_na_xy.shape[0])
    nz = int(depth_values_um.size)
    fx = torch.as_tensor(frequency_x[None, :], dtype=real_dtype, device=device)
    fy = torch.as_tensor(frequency_y[:, None], dtype=real_dtype, device=device)
    source = torch.as_tensor(source_na_xy, dtype=real_dtype, device=device)
    depth_values = torch.as_tensor(depth_values_um.astype(np.float32), dtype=real_dtype, device=device)
    k0 = 2.0 * math.pi / float(wavelength_um)
    k_medium = k0 * float(medium_index)
    max_frequency = float(objective_na) / float(wavelength_um)

    uv1 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    uv2 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    green1 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    green2 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    pupil1 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    pupil2 = torch.empty((m, h, w), dtype=real_dtype, device=device)
    source_axial = torch.empty((m,), dtype=real_dtype, device=device)
    sum_ptf = torch.zeros((h, w, nz), dtype=real_dtype, device=device)
    sum_atf = torch.zeros((h, w, nz), dtype=real_dtype, device=device)
    conj_ptf_atf = torch.zeros((h, w, nz), dtype=complex_dtype, device=device)
    conj_atf_ptf = torch.zeros((h, w, nz), dtype=complex_dtype, device=device)

    def axial_green(delta_fx: Any, delta_fy: Any) -> tuple[Any, Any, Any]:
        radial2 = delta_fx * delta_fx + delta_fy * delta_fy
        inside = 1.0 - float(wavelength_um) ** 2 * radial2
        valid = (inside > float(evanescent_eps)) & (radial2 <= max_frequency * max_frequency)
        axial = torch.where(valid, torch.sqrt(torch.clamp(inside, min=0.0)), torch.zeros_like(inside))
        green = torch.where(valid, 1.0 / (k_medium * torch.clamp(axial, min=1e-30)), torch.zeros_like(axial))
        return axial, green, valid.to(dtype=real_dtype)

    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for frame_index in range(m):
            source_fx = source[frame_index, 0] / float(wavelength_um)
            source_fy = source[frame_index, 1] / float(wavelength_um)
            uv1_i, green1_i, pupil1_i = axial_green(fx - source_fx, fy - source_fy)
            uv2_i, green2_i, pupil2_i = axial_green(fx + source_fx, fy + source_fy)
            source_inside = 1.0 - float(wavelength_um) ** 2 * (source_fx * source_fx + source_fy * source_fy)
            source_axial_i = torch.sqrt(torch.clamp(source_inside, min=float(evanescent_eps)))
            uv1[frame_index] = uv1_i
            uv2[frame_index] = uv2_i
            green1[frame_index] = green1_i
            green2[frame_index] = green2_i
            pupil1[frame_index] = pupil1_i
            pupil2[frame_index] = pupil2_i
            source_axial[frame_index] = source_axial_i
            ptf, atf = transfer_from_geometry_torch(
                torch=torch,
                uv1=uv1_i,
                uv2=uv2_i,
                green1=green1_i,
                green2=green2_i,
                pupil1=pupil1_i,
                pupil2=pupil2_i,
                source_axial=source_axial_i,
                depth_values=depth_values,
                dz_um=dz_um,
                wavelength_um=wavelength_um,
                medium_index=medium_index,
                complex_dtype=complex_dtype,
            )
            sum_ptf += torch.real(torch.conj(ptf) * ptf)
            sum_atf += torch.real(torch.conj(atf) * atf)
            conj_ptf_atf += torch.conj(ptf) * atf
            conj_atf_ptf += torch.conj(atf) * ptf
    support_indices = None
    support_uv1 = None
    support_uv2 = None
    support_green1 = None
    support_green2 = None
    support_pupil1 = None
    support_pupil2 = None
    support_conj_ptf = None
    support_conj_atf = None
    support_active_fraction = 1.0
    if rhs_mode == "support":
        support_indices = []
        support_uv1 = []
        support_uv2 = []
        support_green1 = []
        support_green2 = []
        support_pupil1 = []
        support_pupil2 = []
        active_total = 0
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            for frame_index in range(m):
                support = (pupil1[frame_index] != 0) | (pupil2[frame_index] != 0)
                indices = torch.nonzero(support.reshape(-1), as_tuple=False).reshape(-1).contiguous()
                active_total += int(indices.numel())
                support_indices.append(indices)
                support_uv1.append(torch.index_select(uv1[frame_index].reshape(-1), 0, indices).contiguous())
                support_uv2.append(torch.index_select(uv2[frame_index].reshape(-1), 0, indices).contiguous())
                support_green1.append(torch.index_select(green1[frame_index].reshape(-1), 0, indices).contiguous())
                support_green2.append(torch.index_select(green2[frame_index].reshape(-1), 0, indices).contiguous())
                support_pupil1.append(torch.index_select(pupil1[frame_index].reshape(-1), 0, indices).contiguous())
                support_pupil2.append(torch.index_select(pupil2[frame_index].reshape(-1), 0, indices).contiguous())
        support_indices = tuple(support_indices)
        support_uv1 = tuple(support_uv1)
        support_uv2 = tuple(support_uv2)
        support_green1 = tuple(support_green1)
        support_green2 = tuple(support_green2)
        support_pupil1 = tuple(support_pupil1)
        support_pupil2 = tuple(support_pupil2)
        support_active_fraction = float(active_total / max(1, m * h * w))
        if cache_support_transfer:
            conj_ptf_values = []
            conj_atf_values = []
            with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
                for frame_index in range(m):
                    ptf, atf = transfer_from_geometry_torch(
                        torch=torch,
                        uv1=support_uv1[frame_index],
                        uv2=support_uv2[frame_index],
                        green1=support_green1[frame_index],
                        green2=support_green2[frame_index],
                        pupil1=support_pupil1[frame_index],
                        pupil2=support_pupil2[frame_index],
                        source_axial=source_axial[frame_index],
                        depth_values=depth_values,
                        dz_um=dz_um,
                        wavelength_um=wavelength_um,
                        medium_index=medium_index,
                        complex_dtype=complex_dtype,
                    )
                    conj_ptf_values.append(torch.conj(ptf).contiguous())
                    conj_atf_values.append(torch.conj(atf).contiguous())
            support_conj_ptf = tuple(conj_ptf_values)
            support_conj_atf = tuple(conj_atf_values)
            support_uv1 = None
            support_uv2 = None
            support_green1 = None
            support_green2 = None
            support_pupil1 = None
            support_pupil2 = None
        uv1_cache = uv2_cache = green1_cache = green2_cache = pupil1_cache = pupil2_cache = None
    else:
        uv1_cache = uv1
        uv2_cache = uv2
        green1_cache = green1
        green2_cache = green2
        pupil1_cache = pupil1
        pupil2_cache = pupil2
    solve_coeff_v_re_ptf = None
    solve_coeff_v_re_atf = None
    solve_coeff_v_im_ptf = None
    solve_coeff_v_im_atf = None
    solve_denom_abs_min = float("nan")
    solve_denom_abs_max = float("nan")
    solve_denom_floor = float("nan")
    if cache_solve_coeffs:
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            denom = (sum_ptf + float(alpha)) * (sum_atf + float(beta)) - (conj_ptf_atf * conj_atf_ptf)
            denom_abs = torch.abs(denom)
            denom_floor = torch.clamp(torch.max(denom_abs) * 1e-12, min=1e-30)
            denom = torch.where(denom_abs > denom_floor, denom, denom_floor.to(complex_dtype))
            solve_coeff_v_re_ptf = ((sum_atf + float(beta)) / denom).contiguous()
            solve_coeff_v_re_atf = (-conj_ptf_atf / denom).contiguous()
            solve_coeff_v_im_ptf = (-conj_atf_ptf / denom).contiguous()
            solve_coeff_v_im_atf = ((sum_ptf + float(alpha)) / denom).contiguous()
            solve_denom_abs_min = float(torch.min(denom_abs).detach().cpu().item())
            solve_denom_abs_max = float(torch.max(denom_abs).detach().cpu().item())
            solve_denom_floor = float(denom_floor.detach().cpu().item())
    cache_bytes = tensor_bytes(
        uv1_cache,
        uv2_cache,
        green1_cache,
        green2_cache,
        pupil1_cache,
        pupil2_cache,
        source_axial,
        sum_ptf,
        sum_atf,
        conj_ptf_atf,
        conj_atf_ptf,
        support_indices,
        support_uv1,
        support_uv2,
        support_green1,
        support_green2,
        support_pupil1,
        support_pupil2,
        support_conj_ptf,
        support_conj_atf,
        solve_coeff_v_re_ptf,
        solve_coeff_v_re_atf,
        solve_coeff_v_im_ptf,
        solve_coeff_v_im_atf,
    )
    synchronize(torch, device)
    setup_s = time.perf_counter() - start
    return TorchAidtGeometryPlan(
        torch=torch,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        uv1=uv1_cache,
        uv2=uv2_cache,
        green1=green1_cache,
        green2=green2_cache,
        pupil1=pupil1_cache,
        pupil2=pupil2_cache,
        source_axial=source_axial,
        sum_ptf=sum_ptf,
        sum_atf=sum_atf,
        conj_ptf_atf=conj_ptf_atf,
        conj_atf_ptf=conj_atf_ptf,
        rhs_mode=rhs_mode,
        support_indices=support_indices,
        support_uv1=support_uv1,
        support_uv2=support_uv2,
        support_green1=support_green1,
        support_green2=support_green2,
        support_pupil1=support_pupil1,
        support_pupil2=support_pupil2,
        support_conj_ptf=support_conj_ptf,
        support_conj_atf=support_conj_atf,
        support_active_fraction=support_active_fraction,
        cache_support_transfer=bool(cache_support_transfer),
        cache_solve_coeffs=bool(cache_solve_coeffs),
        solve_coeff_v_re_ptf=solve_coeff_v_re_ptf,
        solve_coeff_v_re_atf=solve_coeff_v_re_atf,
        solve_coeff_v_im_ptf=solve_coeff_v_im_ptf,
        solve_coeff_v_im_atf=solve_coeff_v_im_atf,
        solve_denom_abs_min=solve_denom_abs_min,
        solve_denom_abs_max=solve_denom_abs_max,
        solve_denom_floor=solve_denom_floor,
        depth_values=depth_values,
        wavelength_um=float(wavelength_um),
        medium_index=float(medium_index),
        dz_um=float(dz_um),
        setup_s=float(setup_s),
        cache_bytes=cache_bytes,
    )


def solve_torch_frequency_system(
    *,
    torch: Any,
    plan: TorchAidtGeometryPlan,
    conj_ptf_image: Any,
    conj_atf_image: Any,
    alpha: float,
    beta: float,
    fft_norm: str,
    collect_output_stats: bool,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if plan.cache_solve_coeffs:
        v_re_freq = plan.solve_coeff_v_re_ptf * conj_ptf_image + plan.solve_coeff_v_re_atf * conj_atf_image
        v_im_freq = plan.solve_coeff_v_im_ptf * conj_ptf_image + plan.solve_coeff_v_im_atf * conj_atf_image
        denom_abs_min = float(plan.solve_denom_abs_min)
        denom_abs_max = float(plan.solve_denom_abs_max)
        denom_floor_value = float(plan.solve_denom_floor)
    else:
        denom = (plan.sum_ptf + float(alpha)) * (plan.sum_atf + float(beta)) - (
            plan.conj_ptf_atf * plan.conj_atf_ptf
        )
        denom_abs = torch.abs(denom)
        denom_floor = torch.clamp(torch.max(denom_abs) * 1e-12, min=1e-30)
        denom = torch.where(denom_abs > denom_floor, denom, denom_floor.to(plan.complex_dtype))
        v_re_freq = ((plan.sum_atf + float(beta)) * conj_ptf_image - plan.conj_ptf_atf * conj_atf_image) / denom
        v_im_freq = ((plan.sum_ptf + float(alpha)) * conj_atf_image - plan.conj_atf_ptf * conj_ptf_image) / denom
        denom_abs_min = float(torch.min(denom_abs).detach().cpu().item())
        denom_abs_max = float(torch.max(denom_abs).detach().cpu().item())
        denom_floor_value = float(denom_floor.detach().cpu().item())
    v_re = torch.real(centered_ifft2_torch(torch, v_re_freq, norm=fft_norm)).to(dtype=plan.real_dtype)
    v_im = torch.real(centered_ifft2_torch(torch, v_im_freq, norm=fft_norm)).to(dtype=plan.real_dtype)
    n2_plus_v = float(plan.medium_index) ** 2 + v_re
    n_re = torch.sqrt(torch.clamp((n2_plus_v + torch.sqrt(torch.clamp(n2_plus_v * n2_plus_v + v_im * v_im, min=0.0))) / 2.0, min=0.0))
    n_im = v_im / (2.0 * torch.clamp(n_re, min=1e-12))
    stats = {
        "denom_abs_min": denom_abs_min,
        "denom_abs_max": denom_abs_max,
        "denom_floor": denom_floor_value,
    }
    if collect_output_stats:
        stats.update(
            {
                "v_re_min": float(torch.min(v_re).detach().cpu().item()),
                "v_re_max": float(torch.max(v_re).detach().cpu().item()),
                "v_im_min": float(torch.min(v_im).detach().cpu().item()),
                "v_im_max": float(torch.max(v_im).detach().cpu().item()),
                "n_re_min": float(torch.min(n_re).detach().cpu().item()),
                "n_re_max": float(torch.max(n_re).detach().cpu().item()),
                "n_im_min": float(torch.min(n_im).detach().cpu().item()),
                "n_im_max": float(torch.max(n_im).detach().cpu().item()),
            }
        )
    else:
        stats.update(
            {
                "v_re_min": float("nan"),
                "v_re_max": float("nan"),
                "v_im_min": float("nan"),
                "v_im_max": float("nan"),
                "n_re_min": float("nan"),
                "n_re_max": float("nan"),
                "n_im_min": float("nan"),
                "n_im_max": float("nan"),
            }
        )
    return v_re, v_im, n_re, n_im, stats


def run_torch_reconstruction(
    *,
    plan: TorchAidtGeometryPlan,
    data_t: Any,
    alpha: float,
    beta: float,
    fft_norm: str,
    collect_output_stats: bool,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    torch = plan.torch
    device = plan.device
    h, w = int(data_t.shape[1]), int(data_t.shape[2])
    nz = int(plan.sum_ptf.shape[2])
    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        fft_start = time.perf_counter()
        image_fft = centered_fft2_torch(torch, data_t, norm=fft_norm).to(dtype=plan.complex_dtype)
        synchronize(torch, device)
        fft_s = time.perf_counter() - fft_start

        rhs_start = time.perf_counter()
        if plan.rhs_mode == "support":
            conj_ptf_image_flat = torch.zeros((h * w, nz), dtype=plan.complex_dtype, device=device)
            conj_atf_image_flat = torch.zeros((h * w, nz), dtype=plan.complex_dtype, device=device)
            image_fft_flat = image_fft.reshape(int(data_t.shape[0]), h * w)
            for frame_index in range(int(data_t.shape[0])):
                indices = plan.support_indices[frame_index]
                if plan.cache_support_transfer:
                    conj_ptf = plan.support_conj_ptf[frame_index]
                    conj_atf = plan.support_conj_atf[frame_index]
                else:
                    ptf, atf = transfer_from_geometry_torch(
                        torch=torch,
                        uv1=plan.support_uv1[frame_index],
                        uv2=plan.support_uv2[frame_index],
                        green1=plan.support_green1[frame_index],
                        green2=plan.support_green2[frame_index],
                        pupil1=plan.support_pupil1[frame_index],
                        pupil2=plan.support_pupil2[frame_index],
                        source_axial=plan.source_axial[frame_index],
                        depth_values=plan.depth_values,
                        dz_um=plan.dz_um,
                        wavelength_um=plan.wavelength_um,
                        medium_index=plan.medium_index,
                        complex_dtype=plan.complex_dtype,
                    )
                    conj_ptf = torch.conj(ptf)
                    conj_atf = torch.conj(atf)
                frame_fft = torch.index_select(image_fft_flat[frame_index], 0, indices)
                conj_ptf_image_flat.index_add_(0, indices, conj_ptf * frame_fft[:, None])
                conj_atf_image_flat.index_add_(0, indices, conj_atf * frame_fft[:, None])
            conj_ptf_image = conj_ptf_image_flat.reshape(h, w, nz)
            conj_atf_image = conj_atf_image_flat.reshape(h, w, nz)
        else:
            conj_ptf_image = torch.zeros((h, w, nz), dtype=plan.complex_dtype, device=device)
            conj_atf_image = torch.zeros((h, w, nz), dtype=plan.complex_dtype, device=device)
            for frame_index in range(int(data_t.shape[0])):
                ptf, atf = transfer_from_geometry_torch(
                    torch=torch,
                    uv1=plan.uv1[frame_index],
                    uv2=plan.uv2[frame_index],
                    green1=plan.green1[frame_index],
                    green2=plan.green2[frame_index],
                    pupil1=plan.pupil1[frame_index],
                    pupil2=plan.pupil2[frame_index],
                    source_axial=plan.source_axial[frame_index],
                    depth_values=plan.depth_values,
                    dz_um=plan.dz_um,
                    wavelength_um=plan.wavelength_um,
                    medium_index=plan.medium_index,
                    complex_dtype=plan.complex_dtype,
                )
                frame_fft = image_fft[frame_index]
                conj_ptf_image += torch.conj(ptf) * frame_fft[:, :, None]
                conj_atf_image += torch.conj(atf) * frame_fft[:, :, None]
        synchronize(torch, device)
        rhs_s = time.perf_counter() - rhs_start

        solve_start = time.perf_counter()
        v_re, v_im, n_re, n_im, solve_stats = solve_torch_frequency_system(
            torch=torch,
            plan=plan,
            conj_ptf_image=conj_ptf_image,
            conj_atf_image=conj_atf_image,
            alpha=alpha,
            beta=beta,
            fft_norm=fft_norm,
            collect_output_stats=collect_output_stats,
        )
        synchronize(torch, device)
        solve_s = time.perf_counter() - solve_start
    run_s = time.perf_counter() - start
    stats = {
        **solve_stats,
        "run_s": float(run_s),
        "fft_s": float(fft_s),
        "rhs_s": float(rhs_s),
        "solve_s": float(solve_s),
        "cache_mib": float(plan.cache_bytes / (1024.0 * 1024.0)),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0,
        "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0,
    }
    return v_re, v_im, n_re, n_im, stats


def run_cpu_reference(
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
    dtype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    complex_dtype = np.complex64 if dtype == "complex64" else np.complex128
    start = time.perf_counter()
    accumulation = accumulate_transfer_system(
        data=data,
        frequency_x=frequency_x,
        frequency_y=frequency_y,
        source_na_xy=source_na_xy,
        wavelength_um=wavelength_um,
        medium_index=medium_index,
        objective_na=objective_na,
        depth_values_um=depth_values_um,
        dz_um=dz_um,
        fft_norm=fft_norm,
        evanescent_eps=evanescent_eps,
        complex_dtype=complex_dtype,
    )
    v_re, v_im, n_re, n_im, solve_stats = solve_transfer_system(
        accumulation,
        alpha=alpha,
        beta=beta,
        fft_norm=fft_norm,
        medium_index=medium_index,
    )
    return v_re, v_im, n_re, n_im, {
        "run_s": float(time.perf_counter() - start),
        "accumulation_s": float(accumulation.stats["accumulation_s"]),
        "solve_s": float(solve_stats["solve_s"]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Public aIDT Torch GPU geometry-cache benchmark",
        "",
        "This benchmark runs the public aIDT PTF/ATF transfer reconstruction with the geometry-cache path on Torch/CUDA.",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key in (
        "crop_size",
        "processed_shape",
        "depth_count",
        "n_illum",
        "device_name",
        "dtype",
        "fft_norm",
        "rhs_mode",
        "cache_support_transfer",
        "cache_solve_coeffs",
        "collect_output_stats",
        "support_active_fraction",
        "alpha",
        "beta",
    ):
        lines.append(f"| `{key}` | `{summary.get(key)}` |")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| stage | seconds |",
            "| --- | ---: |",
            f"| GPU setup | {summary['gpu_setup_s']:.6f} |",
            f"| GPU run median | {summary['gpu_run_median_s']:.6f} |",
            f"| GPU FFT median | {summary['gpu_fft_median_s']:.6f} |",
            f"| GPU RHS median | {summary['gpu_rhs_median_s']:.6f} |",
            f"| GPU solve median | {summary['gpu_solve_median_s']:.6f} |",
        ]
    )
    if "cpu_reference_run_s" in summary:
        lines.extend(
            [
                f"| CPU streaming reference | {summary['cpu_reference_run_s']:.6f} |",
                "",
                "## Accuracy vs CPU Reference",
                "",
                "| quantity | rel-L2 |",
                "| --- | ---: |",
                f"| `n_re` | {summary['n_re_rel_l2_vs_cpu']:.6g} |",
                f"| `n_im` | {summary['n_im_rel_l2_vs_cpu']:.6g} |",
                f"| `v_re` | {summary['v_re_rel_l2_vs_cpu']:.6g} |",
                f"| `v_im` | {summary['v_im_rel_l2_vs_cpu']:.6g} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| quantity | MiB |",
            "| --- | ---: |",
            f"| geometry cache estimate | {summary['gpu_cache_mib']:.3f} |",
            f"| Torch peak allocated | {summary['torch_peak_allocated_mib']:.3f} |",
            f"| Torch peak reserved | {summary['torch_peak_reserved_mib']:.3f} |",
        ]
    )
    if "cpu_reference_run_s" in summary:
        lines.extend(
            [
                "",
                "## Repeated-Volume Amortization",
                "",
                "These totals use the measured setup and median hot-run values from this benchmark.",
                "",
                "| volumes | CPU streaming total | GPU total including one setup | effective speedup |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        cpu_s = float(summary["cpu_reference_run_s"])
        setup_s = float(summary["gpu_setup_s"])
        run_s = float(summary["gpu_run_median_s"])

        def format_duration(seconds: float) -> str:
            if seconds < 60.0:
                return f"{seconds:.2f} s"
            if seconds < 3600.0:
                return f"{seconds / 60.0:.2f} min"
            return f"{seconds / 3600.0:.2f} h"

        for volumes in (1, 10, 100, 1000):
            cpu_total = cpu_s * volumes
            gpu_total = setup_s + run_s * volumes
            lines.append(
                f"| {volumes} | `{format_duration(cpu_total)}` | `{format_duration(gpu_total)}` | `{cpu_total / gpu_total:.1f}x` |"
            )
        lines.extend(
            [
                "",
                "## Optimization Readout",
                "",
                f"- GPU throughput after setup: `{1.0 / run_s:.2f} volumes/s`.",
                f"- Active support fraction: `{float(summary.get('support_active_fraction', 1.0)):.3f}`.",
                f"- RHS mode: `{summary.get('rhs_mode')}`.",
                f"- Cached support transfer: `{summary.get('cache_support_transfer')}`.",
                f"- Output min/max diagnostics collected: `{summary.get('collect_output_stats')}`.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark is intended for CUDA; pass a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    measured = load_odt_measured_contract(args.contract)
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    prep = preprocess_aidt_data(
        measured,
        crop_size=None if args.crop_size <= 0 else args.crop_size,
        background_mode=args.background_mode,
        flatfield_mode=args.flatfield_mode,
        frame_scale_mode=args.frame_scale_mode,
    )
    data = prep.data
    source_na_xy = np.asarray(measured["source_na_xy"], dtype=np.float64)
    if args.max_frames is not None:
        data = data[: int(args.max_frames)]
        source_na_xy = source_na_xy[: int(args.max_frames)]
    depth_values_um = make_depth_values(args.depth_min_um, args.depth_max_um, args.depth_step_um)
    complex_dtype, real_dtype, _, _ = torch_dtypes(torch, args.dtype)
    data_t = torch.as_tensor(np.ascontiguousarray(data), dtype=real_dtype, device=device)

    plan = build_torch_geometry_plan(
        torch=torch,
        device=device,
        frequency_x=prep.frequency_x,
        frequency_y=prep.frequency_y,
        source_na_xy=source_na_xy,
        wavelength_um=float(measured["wavelength"]),
        medium_index=float(measured["medium_index"]),
        objective_na=float(measured["objective_na"]),
        depth_values_um=depth_values_um,
        dz_um=float(args.depth_step_um),
        evanescent_eps=float(args.evanescent_eps),
        dtype=args.dtype,
        rhs_mode=args.rhs_mode,
        alpha=float(args.alpha),
        beta=float(args.beta),
        cache_support_transfer=bool(args.cache_support_transfer),
        cache_solve_coeffs=bool(args.cache_solve_coeffs),
    )
    run_stats: list[dict[str, Any]] = []
    outputs = None
    for repeat in range(max(1, int(args.repeats))):
        v_re, v_im, n_re, n_im, stats = run_torch_reconstruction(
            plan=plan,
            data_t=data_t,
            alpha=float(args.alpha),
            beta=float(args.beta),
            fft_norm=args.fft_norm,
            collect_output_stats=not bool(args.skip_output_stats),
        )
        outputs = (v_re, v_im, n_re, n_im)
        stats["repeat"] = int(repeat)
        run_stats.append(stats)
    if outputs is None:
        raise RuntimeError("GPU reconstruction did not run")

    active = run_stats
    summary: dict[str, Any] = {
        "contract": str(args.contract.resolve()),
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "fft_norm": args.fft_norm,
        "crop_size": int(data.shape[1]),
        "processed_shape": [int(v) for v in data.shape],
        "depth_count": int(depth_values_um.size),
        "n_illum": int(data.shape[0]),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "gpu_setup_s": float(plan.setup_s),
        "rhs_mode": str(plan.rhs_mode),
        "cache_support_transfer": bool(plan.cache_support_transfer),
        "cache_solve_coeffs": bool(plan.cache_solve_coeffs),
        "collect_output_stats": not bool(args.skip_output_stats),
        "support_active_fraction": float(plan.support_active_fraction),
        "gpu_run_median_s": float(median(row["run_s"] for row in active)),
        "gpu_fft_median_s": float(median(row["fft_s"] for row in active)),
        "gpu_rhs_median_s": float(median(row["rhs_s"] for row in active)),
        "gpu_solve_median_s": float(median(row["solve_s"] for row in active)),
        "gpu_cache_mib": float(plan.cache_bytes / (1024.0 * 1024.0)),
        "torch_peak_allocated_mib": float(max(row["torch_peak_allocated_mib"] for row in active)),
        "torch_peak_reserved_mib": float(max(row["torch_peak_reserved_mib"] for row in active)),
        "n_re_min": float(active[-1]["n_re_min"]),
        "n_re_max": float(active[-1]["n_re_max"]),
        "n_im_min": float(active[-1]["n_im_min"]),
        "n_im_max": float(active[-1]["n_im_max"]),
        "validation": report.to_dict(),
        "summary_md": str(args.summary_md.resolve()) if args.summary_md else None,
    }
    if args.compare_cpu:
        cpu_v_re, cpu_v_im, cpu_n_re, cpu_n_im, cpu_stats = run_cpu_reference(
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
            dtype=args.dtype,
        )
        gpu_v_re = outputs[0].detach().cpu().numpy()
        gpu_v_im = outputs[1].detach().cpu().numpy()
        gpu_n_re = outputs[2].detach().cpu().numpy()
        gpu_n_im = outputs[3].detach().cpu().numpy()
        summary.update(
            {
                "cpu_reference_run_s": float(cpu_stats["run_s"]),
                "cpu_reference_accumulation_s": float(cpu_stats["accumulation_s"]),
                "cpu_reference_solve_s": float(cpu_stats["solve_s"]),
                "gpu_run_speedup_vs_cpu_streaming": float(cpu_stats["run_s"] / summary["gpu_run_median_s"]),
                "gpu_setup_plus_run_speedup_vs_cpu_streaming": float(cpu_stats["run_s"] / (summary["gpu_setup_s"] + summary["gpu_run_median_s"])),
                "v_re_rel_l2_vs_cpu": rel_l2(gpu_v_re, cpu_v_re),
                "v_im_rel_l2_vs_cpu": rel_l2(gpu_v_im, cpu_v_im),
                "n_re_rel_l2_vs_cpu": rel_l2(gpu_n_re, cpu_n_re),
                "n_im_rel_l2_vs_cpu": rel_l2(gpu_n_im, cpu_n_im),
            }
        )
    payload = {"config": vars(args), "summary": summary, "runs": run_stats}
    write_json(args.out, payload)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run public aIDT geometry-cache transfer reconstruction on Torch/CUDA.")
    p.add_argument("--contract", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--depth-min-um", type=float, default=-25.5)
    p.add_argument("--depth-max-um", type=float, default=25.5)
    p.add_argument("--depth-step-um", type=float, default=1.5)
    p.add_argument("--alpha", type=float, default=1e2)
    p.add_argument("--beta", type=float, default=1e2)
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--rhs-mode", choices=["dense", "support"], default="dense")
    p.add_argument("--cache-support-transfer", action="store_true")
    p.add_argument("--cache-solve-coeffs", action="store_true")
    p.add_argument("--skip-output-stats", action="store_true")
    p.add_argument("--fft-norm", choices=["none", "ortho"], default="ortho")
    p.add_argument("--background-mode", choices=["none", "frame_mean", "global_mean", "edge_median"], default="frame_mean")
    p.add_argument("--flatfield-mode", choices=["none", "global_mean"], default="none")
    p.add_argument("--frame-scale-mode", choices=["none", "rms"], default="none")
    p.add_argument("--evanescent-eps", type=float, default=1e-9)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--compare-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_torch_gpu.json")
    p.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_public_transfer_torch_gpu.md")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
