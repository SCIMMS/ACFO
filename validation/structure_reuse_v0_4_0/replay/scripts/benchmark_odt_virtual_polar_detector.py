from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    resolve_device,
    synchronize,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    cufinufft_forward,
    make_cufinufft_composite,
)
from benchmark_odt_cone_illumination import cone_illumination_directions  # noqa: E402
from benchmark_odt_ewald_cap_operator import (  # noqa: E402
    QSamples,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    CompositeContext,
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import parser as odt_base_parser  # noqa: E402


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_128 = np.asarray(candidate, dtype=np.complex128)
    reference_128 = np.asarray(reference, dtype=np.complex128)
    denominator = max(float(np.linalg.norm(reference_128.ravel())), 1e-30)
    return float(np.linalg.norm((candidate_128 - reference_128).ravel()) / denominator)


def timing_summary(times: list[float]) -> dict[str, float | int]:
    values = np.asarray(times, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_s": float(np.median(values)),
        "mean_s": float(values.mean()),
        "p05_s": float(np.percentile(values, 5)),
        "p95_s": float(np.percentile(values, 95)),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
    }


class CachedBilinearPolarRemap:
    """Prepared Cartesian-to-polar bilinear gather with an exact scatter adjoint."""

    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        n_xy: int,
        n_radial: int,
        n_phi: int,
        complex_dtype: Any,
        radial_fraction: float = 1.0,
        radial_nodes_fraction: np.ndarray | None = None,
        normalize_pupil_boundary: bool = True,
    ) -> None:
        if n_xy < 2 or n_radial <= 0 or n_phi <= 0:
            raise ValueError("invalid Cartesian or polar grid size")
        if radial_fraction <= 0.0 or radial_fraction > 1.0:
            raise ValueError("radial_fraction must be in (0, 1]")
        self.torch = torch
        self.device = device
        self.n_xy = int(n_xy)
        self.n_radial = int(n_radial)
        self.n_phi = int(n_phi)
        self.complex_dtype = complex_dtype
        self.real_dtype = (
            torch.float32 if complex_dtype == torch.complex64 else torch.float64
        )
        self.radial_fraction = float(radial_fraction)
        self.normalize_pupil_boundary = bool(normalize_pupil_boundary)

        if radial_nodes_fraction is None:
            radial = (
                (np.arange(self.n_radial, dtype=np.float64) + 0.5)
                / float(self.n_radial)
                * self.radial_fraction
            )
        else:
            radial = np.asarray(radial_nodes_fraction, dtype=np.float64)
            if radial.shape != (self.n_radial,):
                raise ValueError(
                    "radial_nodes_fraction must have shape (n_radial,)"
                )
            if (
                np.any(radial <= 0.0)
                or np.any(radial > 1.0)
                or np.any(np.diff(radial) <= 0.0)
            ):
                raise ValueError(
                    "radial_nodes_fraction must be strictly increasing in (0, 1]"
                )
            radial = np.ascontiguousarray(radial)
        phi = np.linspace(0.0, 2.0 * np.pi, self.n_phi, endpoint=False)
        rr, pp = np.meshgrid(radial, phi, indexing="ij")
        x_norm = rr * np.cos(pp)
        y_norm = rr * np.sin(pp)
        x_index = (x_norm + 1.0) * 0.5 * float(self.n_xy - 1)
        y_index = (y_norm + 1.0) * 0.5 * float(self.n_xy - 1)
        x0 = np.floor(x_index).astype(np.int64)
        y0 = np.floor(y_index).astype(np.int64)
        x1 = np.minimum(x0 + 1, self.n_xy - 1)
        y1 = np.minimum(y0 + 1, self.n_xy - 1)
        wx = x_index - x0
        wy = y_index - y0
        indices = np.stack(
            [
                y0 * self.n_xy + x0,
                y0 * self.n_xy + x1,
                y1 * self.n_xy + x0,
                y1 * self.n_xy + x1,
            ],
            axis=0,
        ).reshape(4, -1)
        weights = np.stack(
            [
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ],
            axis=0,
        ).reshape(4, -1)
        axis = np.linspace(-1.0, 1.0, self.n_xy, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        pupil_mask = xx**2 + yy**2 <= 1.0 + 1e-14

        pupil_flat = pupil_mask.ravel()
        valid_weights = weights * pupil_flat[indices]
        pupil_normalization = valid_weights.sum(axis=0)
        if np.any(pupil_normalization <= 0.0):
            raise RuntimeError("polar target has no supporting Cartesian pupil pixels")
        if self.normalize_pupil_boundary:
            weights = valid_weights / pupil_normalization.reshape(1, -1)

        self.grid = torch.as_tensor(
            np.stack([x_norm, y_norm], axis=-1)[None, ...],
            dtype=self.real_dtype,
            device=device,
        )
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=device)
        self.weights = torch.as_tensor(
            weights, dtype=self.real_dtype, device=device
        )
        self.pupil_mask = torch.as_tensor(
            pupil_mask, dtype=torch.bool, device=device
        )
        self.pupil_normalization = torch.as_tensor(
            pupil_normalization.reshape(self.n_radial, self.n_phi),
            dtype=self.real_dtype,
            device=device,
        )
        self.radial = radial

    @property
    def target_count(self) -> int:
        return self.n_radial * self.n_phi

    @property
    def active_fraction(self) -> float:
        return float(self.pupil_mask.sum().item() / self.pupil_mask.numel())

    @property
    def cache_bytes(self) -> int:
        tensors = (
            self.grid,
            self.indices,
            self.weights,
            self.pupil_mask,
            self.pupil_normalization,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def _validate_camera(self, camera: Any) -> None:
        if camera.ndim != 3 or tuple(camera.shape[1:]) != (
            self.n_xy,
            self.n_xy,
        ):
            raise ValueError("camera must have shape (batch, n_xy, n_xy)")
        if camera.dtype != self.complex_dtype:
            raise ValueError("camera complex dtype does not match remap plan")

    def _validate_polar(self, polar: Any) -> None:
        if polar.ndim != 3 or tuple(polar.shape[1:]) != (
            self.n_radial,
            self.n_phi,
        ):
            raise ValueError("polar must have shape (batch, n_radial, n_phi)")
        if polar.dtype != self.complex_dtype:
            raise ValueError("polar complex dtype does not match remap plan")

    def gather_explicit(self, camera: Any, *, batch_block: int = 0) -> Any:
        self._validate_camera(camera)
        block = camera.shape[0] if batch_block <= 0 else int(batch_block)
        output = self.torch.empty(
            (camera.shape[0], self.n_radial, self.n_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        weights = self.weights
        for start in range(0, camera.shape[0], block):
            stop = min(start + block, camera.shape[0])
            flat = camera[start:stop].reshape(stop - start, -1)
            gathered = flat.index_select(1, self.indices[0]) * weights[0]
            for corner in range(1, 4):
                gathered = gathered + flat.index_select(
                    1, self.indices[corner]
                ) * weights[corner]
            output[start:stop] = gathered.reshape(
                stop - start, self.n_radial, self.n_phi
            )
        return output

    def adjoint_explicit(self, polar: Any, *, batch_block: int = 0) -> Any:
        self._validate_polar(polar)
        block = polar.shape[0] if batch_block <= 0 else int(batch_block)
        output = self.torch.zeros(
            (polar.shape[0], self.n_xy, self.n_xy),
            dtype=self.complex_dtype,
            device=self.device,
        )
        weights = self.weights
        for start in range(0, polar.shape[0], block):
            stop = min(start + block, polar.shape[0])
            values = polar[start:stop].reshape(stop - start, -1)
            flat = output[start:stop].reshape(stop - start, -1)
            for corner in range(4):
                flat.scatter_add_(
                    1,
                    self.indices[corner].reshape(1, -1).expand(stop - start, -1),
                    values * weights[corner],
                )
        return output

    def gather_grid_sample(self, camera: Any, *, batch_block: int = 0) -> Any:
        """Low-level CUDA hot path; algebraically identical to explicit gather."""
        self._validate_camera(camera)
        block = camera.shape[0] if batch_block <= 0 else int(batch_block)
        output = self.torch.empty(
            (camera.shape[0], self.n_radial, self.n_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, camera.shape[0], block):
            stop = min(start + block, camera.shape[0])
            channels = self.torch.view_as_real(camera[start:stop]).permute(
                0, 3, 1, 2
            )
            sampled_channels = self.torch.nn.functional.grid_sample(
                channels,
                self.grid.expand(stop - start, -1, -1, -1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            if self.normalize_pupil_boundary:
                sampled_channels = sampled_channels / self.pupil_normalization
            output[start:stop] = self.torch.view_as_complex(
                sampled_channels.permute(0, 2, 3, 1).contiguous()
            )
        return output


def cartesian_q_samples(
    *,
    k: float,
    detector_na: float,
    n_xy: int,
    illumination: np.ndarray,
) -> tuple[QSamples, np.ndarray, np.ndarray]:
    axis = np.linspace(-detector_na, detector_na, n_xy, dtype=np.float64)
    sx, sy = np.meshgrid(axis, axis, indexing="xy")
    mask = sx**2 + sy**2 <= detector_na**2 + 1e-14
    active = np.flatnonzero(mask.ravel())
    sx_active = sx.ravel()[active]
    sy_active = sy.ravel()[active]
    sz_active = np.sqrt(
        np.maximum(1.0 - sx_active**2 - sy_active**2, 0.0)
    )
    detector = np.column_stack([sx_active, sy_active, sz_active])
    illumination = np.asarray(illumination, dtype=np.float64)
    if illumination.ndim != 2 or illumination.shape[1] != 3:
        raise ValueError("illumination must have shape (n_illum, 3)")
    n_illum = int(illumination.shape[0])
    blocks = [k * (detector - s_in[None, :]) for s_in in illumination]
    q = np.vstack(blocks)
    illum_index = np.concatenate(
        [np.full(detector.shape[0], i, dtype=np.int64) for i in range(n_illum)]
    )
    return (
        QSamples(
            qx=np.ascontiguousarray(q[:, 0]),
            qy=np.ascontiguousarray(q[:, 1]),
            qz=np.ascontiguousarray(q[:, 2]),
            q_perp=np.ascontiguousarray(np.hypot(q[:, 0], q[:, 1])),
            phi=np.ascontiguousarray(
                np.mod(np.arctan2(q[:, 1], q[:, 0]), 2.0 * np.pi)
            ),
            illumination_index=np.ascontiguousarray(illum_index),
        ),
        mask,
        active,
    )


def remap_sanity(torch: Any, device: Any, dtype: str) -> dict[str, float]:
    complex_dtype = torch.complex64 if dtype == "complex64" else torch.complex128
    plan = CachedBilinearPolarRemap(
        torch=torch,
        device=device,
        n_xy=33,
        n_radial=12,
        n_phi=32,
        complex_dtype=complex_dtype,
        radial_fraction=0.9,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    real_dtype = torch.float32 if dtype == "complex64" else torch.float64
    camera = torch.complex(
        torch.randn((3, 33, 33), dtype=real_dtype, device=device, generator=generator),
        torch.randn((3, 33, 33), dtype=real_dtype, device=device, generator=generator),
    )
    camera.mul_(plan.pupil_mask)
    residual = torch.complex(
        torch.randn((3, 12, 32), dtype=real_dtype, device=device, generator=generator),
        torch.randn((3, 12, 32), dtype=real_dtype, device=device, generator=generator),
    )
    explicit = plan.gather_explicit(camera)
    fused = plan.gather_grid_sample(camera)
    adjoint = plan.adjoint_explicit(residual)
    lhs = torch.vdot(explicit.reshape(-1), residual.reshape(-1))
    rhs = torch.vdot(camera.reshape(-1), adjoint.reshape(-1))
    dot_error = float(
        (
            torch.abs(lhs - rhs)
            / torch.clamp(torch.abs(lhs) + torch.abs(rhs), min=1e-30)
        )
        .detach()
        .cpu()
        .item()
    )
    fused_error = float(
        (
            torch.linalg.vector_norm(fused - explicit)
            / torch.clamp(torch.linalg.vector_norm(explicit), min=1e-30)
        )
        .detach()
        .cpu()
        .item()
    )
    return {"dot_error": dot_error, "grid_sample_rel_l2": fused_error}


def run_hot_sweep(args: Any, torch: Any, device: Any) -> list[dict[str, Any]]:
    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    real_dtype = torch.float32 if args.dtype == "complex64" else torch.float64
    rows: list[dict[str, Any]] = []
    for n_xy in args.camera_sizes:
        torch.cuda.empty_cache()
        setup_start = time.perf_counter()
        plan = CachedBilinearPolarRemap(
            torch=torch,
            device=device,
            n_xy=n_xy,
            n_radial=args.polar_radial,
            n_phi=args.polar_phi,
            complex_dtype=complex_dtype,
            radial_fraction=args.radial_fraction,
            normalize_pupil_boundary=args.normalize_pupil_boundary,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(1000 + int(n_xy))
        camera = torch.complex(
            torch.randn(
                (args.n_illum, n_xy, n_xy),
                dtype=real_dtype,
                device=device,
                generator=generator,
            ),
            torch.randn(
                (args.n_illum, n_xy, n_xy),
                dtype=real_dtype,
                device=device,
                generator=generator,
            ),
        )
        camera.mul_(plan.pupil_mask)
        synchronize(torch, device)
        setup_s = time.perf_counter() - setup_start

        value = None
        for _ in range(args.warmups):
            value = plan.gather_grid_sample(camera, batch_block=args.batch_block)
            synchronize(torch, device)
        del value
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for _ in range(args.repeats):
            synchronize(torch, device)
            start = time.perf_counter()
            value = plan.gather_grid_sample(camera, batch_block=args.batch_block)
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
        summary = timing_summary(times)
        rows.append(
            {
                "camera_n_xy": int(n_xy),
                "camera_pixels_per_view": int(n_xy * n_xy),
                "camera_active_pixels_per_view": int(plan.pupil_mask.sum().item()),
                "camera_active_fraction": plan.active_fraction,
                "polar_samples_per_view": int(plan.target_count),
                "n_illum": int(args.n_illum),
                "setup_s": float(setup_s),
                "cache_mib": float(plan.cache_bytes / 1024**2),
                "camera_input_mib": float(camera.numel() * camera.element_size() / 1024**2),
                "polar_output_mib": float(value.numel() * value.element_size() / 1024**2),
                "hot_peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "hot_timing": summary,
                "hot_rate_hz": float(1.0 / summary["median_s"]),
            }
        )
        del value, camera, plan
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def accuracy_context(args: Any) -> tuple[Any, CompositeContext]:
    odt_args = odt_base_parser().parse_args([])
    updates = {
        "device": "cuda",
        "dtype": args.dtype,
        "n_r": args.accuracy_object_r,
        "n_z": args.accuracy_object_z,
        "n_beta": args.accuracy_object_beta,
        "r_max": 1.0,
        "z_max": 0.8,
        "phantom": "random_beads",
        "seed": 123,
        "k": args.k,
        "detector_na": args.detector_na,
        "illumination_angle_deg": args.illumination_angle_deg,
        "ring_illum": args.accuracy_ring_illum,
        "skip_axis_illumination": False,
        "cap_radial": args.accuracy_polar_radial,
        "cap_phi": args.accuracy_polar_phi,
        "h_margin": 8,
        "l_margin": 6,
        "cone_l_prune_threshold": 1e-12,
        "cpp_threads": 4,
        "skip_native_prepared_adjoint": True,
        "low_memory_adjoint": True,
        "compact_axisymmetric_kernel": True,
        "real_object": True,
        "radial_block_size": 8,
        "illumination_block_size": 4,
    }
    for name, value in updates.items():
        setattr(odt_args, name, value)
    return odt_args, build_composite_context(odt_args)


def cartesian_context(
    odt_args: Any, ctx: CompositeContext, n_xy: int
) -> tuple[CompositeContext, np.ndarray, np.ndarray]:
    ring_na = math.sin(math.radians(float(odt_args.illumination_angle_deg)))
    ring_illumination = cone_illumination_directions(
        n_illum=odt_args.ring_illum,
        illumination_na=ring_na,
    )[0]
    ring_q, mask, active = cartesian_q_samples(
        k=odt_args.k,
        detector_na=odt_args.detector_na,
        n_xy=n_xy,
        illumination=ring_illumination,
    )
    ring = replace(ctx.ring, flat_q=ring_q)
    axis = None
    if ctx.axis is not None:
        axis_q, axis_mask, axis_active = cartesian_q_samples(
            k=odt_args.k,
            detector_na=odt_args.detector_na,
            n_xy=n_xy,
            illumination=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )
        if not np.array_equal(mask, axis_mask) or not np.array_equal(
            active, axis_active
        ):
            raise RuntimeError("ring and axis Cartesian detector masks differ")
        axis = replace(ctx.axis, flat_q=axis_q)
    return CompositeContext(ring=ring, axis=axis), mask, active


def run_physics_accuracy(
    args: Any, torch: Any, device: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    odt_args, ctx = accuracy_context(args)
    polar_op = make_cufinufft_composite(
        ctx, dtype=args.dtype, plan_mode="plan", eps=args.cufinufft_eps
    )
    cp = polar_op.cp
    np_complex = np.complex64 if args.dtype == "complex64" else np.complex128
    coeff_np = np.ascontiguousarray(
        np.real(ctx.ring.obj.coeff).astype(np_complex, copy=False).ravel()
    )
    coeff = cp.asarray(coeff_np)
    polar_parts_cp = cufinufft_forward(
        polar_op, coeff, eps=args.cufinufft_eps
    )
    cp.cuda.get_current_stream().synchronize()
    polar_parts = [np.asarray(part.get()) for part in polar_parts_cp]

    complex_dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    rows: list[dict[str, Any]] = []
    for n_xy in args.accuracy_camera_sizes:
        cart_ctx, mask, active = cartesian_context(odt_args, ctx, n_xy)
        cart_op = make_cufinufft_composite(
            cart_ctx, dtype=args.dtype, plan_mode="plan", eps=args.cufinufft_eps
        )
        cart_parts_cp = cufinufft_forward(
            cart_op, coeff, eps=args.cufinufft_eps
        )
        cp.cuda.get_current_stream().synchronize()
        cart_parts = [np.asarray(part.get()) for part in cart_parts_cp]
        plan = CachedBilinearPolarRemap(
            torch=torch,
            device=device,
            n_xy=n_xy,
            n_radial=args.accuracy_polar_radial,
            n_phi=args.accuracy_polar_phi,
            complex_dtype=complex_dtype,
            radial_fraction=1.0,
            normalize_pupil_boundary=args.normalize_pupil_boundary,
        )
        remapped_parts: list[np.ndarray] = []
        reference_parts: list[np.ndarray] = []
        inner_remapped_parts: list[np.ndarray] = []
        inner_reference_parts: list[np.ndarray] = []
        for part_index, (cart_values, polar_values) in enumerate(
            zip(cart_parts, polar_parts)
        ):
            n_illum = (
                odt_args.ring_illum if part_index == 0 else 1
            )
            frames = np.zeros((n_illum, n_xy * n_xy), dtype=np_complex)
            frames[:, active] = cart_values.reshape(n_illum, active.size)
            camera = torch.as_tensor(
                frames.reshape(n_illum, n_xy, n_xy),
                dtype=complex_dtype,
                device=device,
            )
            remapped = (
                plan.gather_grid_sample(camera, batch_block=args.batch_block)
                .detach()
                .cpu()
                .numpy()
            )
            reference = polar_values.reshape(
                n_illum, args.accuracy_polar_radial, args.accuracy_polar_phi
            )
            remapped_parts.append(remapped.ravel())
            reference_parts.append(reference.ravel())
            inner_rows = plan.radial <= args.accuracy_inner_fraction
            inner_remapped_parts.append(remapped[:, inner_rows, :].ravel())
            inner_reference_parts.append(reference[:, inner_rows, :].ravel())
        remapped_all = np.concatenate(remapped_parts)
        reference_all = np.concatenate(reference_parts)
        inner_remapped = np.concatenate(inner_remapped_parts)
        inner_reference = np.concatenate(inner_reference_parts)
        rows.append(
            {
                "camera_n_xy": int(n_xy),
                "equivalent_full_camera_n_xy": int(
                    round(
                        n_xy
                        * args.polar_radial
                        / args.accuracy_polar_radial
                    )
                ),
                "active_pixels_per_view": int(active.size),
                "active_to_polar_sample_ratio": float(
                    active.size
                    / (args.accuracy_polar_radial * args.accuracy_polar_phi)
                ),
                "full_pupil_rel_l2": relative_l2(remapped_all, reference_all),
                "inner_fraction": float(args.accuracy_inner_fraction),
                "inner_pupil_rel_l2": relative_l2(
                    inner_remapped, inner_reference
                ),
            }
        )
        del cart_op, cart_parts_cp, cart_parts, plan
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    metadata = {
        "object_shape": list(ctx.ring.obj.coeff.shape),
        "polar_shape_per_view": [
            args.accuracy_polar_radial,
            args.accuracy_polar_phi,
        ],
        "ring_illum": int(odt_args.ring_illum),
        "axis_illumination_included": ctx.axis is not None,
        "q_reference": "cuFINUFFT type-3 evaluated independently at Cartesian and polar detector directions",
        "cufinufft_eps": float(args.cufinufft_eps),
    }
    return metadata, rows


def load_acfo_selected_slice_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"n_selected_{int(row['selected_n_z'])}": float(row["acfo_pair_median_s"])
        for row in payload.get("rows", [])
    }


def write_outputs(args: Any, result: dict[str, Any]) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "camera_n_xy",
            "camera_pixels_per_view",
            "camera_active_pixels_per_view",
            "camera_active_fraction",
            "polar_samples_per_view",
            "n_illum",
            "cache_mib",
            "camera_input_mib",
            "polar_output_mib",
            "hot_peak_allocated_mib",
            "hot_median_ms",
            "hot_rate_hz",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["hot_sweep"]:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "hot_median_ms": 1000.0 * row["hot_timing"]["median_s"],
                }
            )

    lines = [
        "# ODT cached virtual polar detector 검증",
        "",
        "## Hot GPU remap",
        "",
        f"GPU-resident complex Cartesian frames {args.n_illum}개를 "
        f"{args.polar_radial}×{args.polar_phi} polar cap으로 변환했다. "
        "고정 grid 준비와 데이터 전송은 제외했다.",
        "",
        "| Cartesian camera | active/view | input MiB | remap median ms | remap Hz | peak MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["hot_sweep"]:
        lines.append(
            f"| {row['camera_n_xy']}² | {row['camera_active_pixels_per_view']:,} | "
            f"{row['camera_input_mib']:.1f} | "
            f"{1000.0 * row['hot_timing']['median_s']:.3f} | "
            f"{row['hot_rate_hz']:.2f} | {row['hot_peak_allocated_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 물리 reference 정확도",
            "",
            "cuFINUFFT type-3로 동일한 작은 ODT 물체를 Cartesian detector 좌표와 이상적 polar detector 좌표에서 각각 직접 계산했다.",
            "",
            "| Accuracy camera | equivalent full camera | active/polar ratio | full-pupil rel-L2 | inner-pupil rel-L2 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["physics_accuracy"]["rows"]:
        lines.append(
            f"| {row['camera_n_xy']}² | {row['equivalent_full_camera_n_xy']}² | "
            f"{row['active_to_polar_sample_ratio']:.3f} | "
            f"{row['full_pupil_rel_l2']:.3g} | {row['inner_pupil_rel_l2']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- 이 remap은 acquisition마다 한 번 수행하는 전처리다. 반복 reconstruction에서는 같은 polar data를 재사용할 수 있다.",
            "- 시간에는 GPU 전송과 image-plane hologram 복원/FFT가 포함되지 않는다.",
            "- full-pupil 오차에는 원형 pupil 경계의 불연속을 bilinear interpolation하는 오차가 포함된다.",
            "- polar-space residual을 푸는 경우의 전처리이며, Cartesian detector-space likelihood를 정확히 유지하려면 remap의 forward/adjoint를 iteration 안에 포함해야 한다.",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark a cached Cartesian-camera to virtual-polar ODT detector."
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--camera-sizes", nargs="+", type=int, default=[256, 320, 384, 512])
    p.add_argument("--n-illum", type=int, default=121)
    p.add_argument("--polar-radial", type=int, default=256)
    p.add_argument("--polar-phi", type=int, default=256)
    p.add_argument("--radial-fraction", type=float, default=1.0)
    p.add_argument(
        "--no-pupil-normalization",
        action="store_false",
        dest="normalize_pupil_boundary",
    )
    p.add_argument("--batch-block", type=int, default=16)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument(
        "--accuracy-camera-sizes", nargs="+", type=int, default=[64, 80, 96, 128]
    )
    p.add_argument("--accuracy-polar-radial", type=int, default=64)
    p.add_argument("--accuracy-polar-phi", type=int, default=64)
    p.add_argument("--accuracy-inner-fraction", type=float, default=0.95)
    p.add_argument("--accuracy-object-r", type=int, default=32)
    p.add_argument("--accuracy-object-z", type=int, default=16)
    p.add_argument("--accuracy-object-beta", type=int, default=64)
    p.add_argument("--accuracy-ring-illum", type=int, default=4)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument(
        "--acfo-selected-summary",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_selected_z_gpu_hot_sweep_summary.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_gpu.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_gpu.csv",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT
        / "benchmark_results"
        / "odt_virtual_polar_detector_gpu_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    sanity = remap_sanity(torch, device, args.dtype)
    hot_rows = run_hot_sweep(args, torch, device)
    accuracy_metadata, accuracy_rows = run_physics_accuracy(
        args, torch, device
    )
    selected_times = load_acfo_selected_slice_times(args.acfo_selected_summary)
    for row in hot_rows:
        remap_s = float(row["hot_timing"]["median_s"])
        row["selected_slice_pipeline_estimates"] = {
            key: {
                "acfo_pair_s": acfo_s,
                "remap_plus_one_pair_s": remap_s + acfo_s,
                "remap_overhead_fraction_of_pair": remap_s / acfo_s,
                "one_pair_pipeline_hz": 1.0 / (remap_s + acfo_s),
            }
            for key, acfo_s in selected_times.items()
            if key in {"n_selected_1", "n_selected_8"}
        }
    result = {
        "schema": "odt-virtual-polar-detector-gpu-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "protocol": {
            "full_measurement_frames": int(args.n_illum),
            "polar_shape_per_view": [args.polar_radial, args.polar_phi],
            "data_already_gpu_resident": True,
            "cached_geometry_setup_excluded": True,
            "timed_path": "complex input -> fused CUDA bilinear grid_sample -> complex polar output",
            "batch_block": int(args.batch_block),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "pupil_boundary_weight_normalization": bool(
                args.normalize_pupil_boundary
            ),
        },
        "operator_sanity": sanity,
        "hot_sweep": hot_rows,
        "physics_accuracy": {
            "metadata": accuracy_metadata,
            "rows": accuracy_rows,
        },
        "claim_boundary": [
            "The virtual polar remap is performed once per acquired complex-field stack and can be reused across reconstruction iterations.",
            "GPU transfer, hologram demodulation, and any image-plane FFT are excluded.",
            "The physical accuracy test independently evaluates the same object at Cartesian and polar detector directions with cuFINUFFT type-3.",
            "A polar-space likelihood is not identical to an exact Cartesian detector-space likelihood unless the remap operator and its adjoint are included in the model.",
        ],
    }
    write_outputs(args, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
