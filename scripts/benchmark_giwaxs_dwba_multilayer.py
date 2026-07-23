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
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_giwaxs_prepared_operator import (  # noqa: E402
    GixsDetector,
    _build_mode_cutoffs,
    _cpp_solvers_module,
    _group_cutoffs_by_qz,
    _mode_indices,
    _positive_max_mode,
    make_giwaxs_detector,
    summarize_detector,
)
from waxs_cake import make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


@dataclass(frozen=True)
class DistortedWaveStack:
    """Layer-wise distorted-field coefficients supplied to the scattering sum."""

    z_edges_nm: np.ndarray
    kz_incident: np.ndarray
    incident_forward: np.ndarray
    incident_reflected: np.ndarray
    kz_exit: np.ndarray
    exit_forward: np.ndarray
    exit_reflected: np.ndarray
    alpha_incident: np.ndarray
    beta_incident: np.ndarray
    alpha_exit: np.ndarray
    beta_exit: np.ndarray
    incident_angle_deg: float
    critical_angle_deg: np.ndarray
    target_alpha_f_deg: np.ndarray

    @property
    def n_layers(self) -> int:
        return int(self.z_edges_nm.size - 1)


@dataclass(frozen=True)
class PreparedDwbaGeometry:
    indices: np.ndarray
    modes: np.ndarray
    kernel: np.ndarray
    n_phi: int
    hcoef: np.ndarray | None = None
    cutoffs: np.ndarray | None = None
    requested_max_mode: int = 0
    max_mode: int = 0
    mode_pruning: bool = False
    cutoff_margin: int = 0
    cutoff_bin_size: int = 1
    cutoff_min: int = 0
    cutoff_mean: float = 0.0
    mode_work_fraction: float = 1.0


@dataclass(frozen=True)
class PreparedDwbaMillerGeometry:
    qperp: np.ndarray
    phi: np.ndarray
    r_centers: np.ndarray
    hhat: np.ndarray
    cutoffs: np.ndarray
    n_phi: int
    requested_max_mode: int
    max_mode: int
    extra_order: int
    complex_dtype: np.dtype
    mode_pruning: bool
    cutoff_margin: int
    cutoff_bin_size: int
    cutoff_min: int
    cutoff_mean: float
    mode_work_fraction: float


def median_time(func, repeats: int) -> tuple[Any, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def make_synthetic_multilayer(
    *,
    n_atoms: int,
    n_layers: int,
    radius_nm: float,
    layer_thickness_nm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if radius_nm <= 0.0 or layer_thickness_nm <= 0.0:
        raise ValueError("radius_nm and layer_thickness_nm must be positive")

    rng = np.random.default_rng(seed)
    total_thickness = n_layers * layer_thickness_nm
    z_edges = np.linspace(-0.5 * total_thickness, 0.5 * total_thickness, n_layers + 1)
    counts = np.full(n_layers, n_atoms // n_layers, dtype=np.int64)
    counts[: n_atoms % n_layers] += 1

    coords_parts: list[np.ndarray] = []
    layer_parts: list[np.ndarray] = []
    for layer, count in enumerate(counts):
        theta = rng.uniform(0.0, 2.0 * math.pi, size=count)
        radius = radius_nm * np.sqrt(rng.uniform(0.0, 1.0, size=count))
        z = rng.uniform(z_edges[layer], z_edges[layer + 1], size=count)
        coords_parts.append(
            np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
        )
        layer_parts.append(np.full(count, layer, dtype=np.int64))

    coords = np.ascontiguousarray(np.vstack(coords_parts), dtype=np.float64)
    layer_ids = np.ascontiguousarray(np.concatenate(layer_parts), dtype=np.int64)
    order = rng.permutation(n_atoms)
    return coords[order], layer_ids[order], z_edges


def _target_alpha_f(detector: GixsDetector) -> np.ndarray:
    alpha = np.broadcast_to(detector.alpha_f_deg[:, None], detector.shape)
    return np.ascontiguousarray(alpha.ravel(), dtype=np.float64)


def build_distorted_wave_stack(
    detector: GixsDetector,
    z_edges_nm: np.ndarray,
    *,
    critical_angle_start_deg: float = 0.13,
    critical_angle_step_deg: float = 0.018,
    absorption_imag: float = 2.5e-4,
    reflectivity_scale: float = 0.28,
    beta_loss_per_layer: float = 0.018,
) -> DistortedWaveStack:
    """Build a propagating-field DWBA-like correction for a fixed multilayer stack.

    This is not a replacement for a Fresnel/Parratt field solver. It assumes the
    layer-wise reflection coefficients (alpha) and transmission factors (beta)
    have already been reduced to arrays, then tests the scattering contraction.
    The speed-comparison use case is the above-critical, real-effective-qz
    regime; evanescent fields are deliberately treated as outside this generic
    FINUFFT-comparable baseline.
    """

    z_edges = np.asarray(z_edges_nm, dtype=np.float64)
    if z_edges.ndim != 1 or z_edges.size < 2:
        raise ValueError("z_edges_nm must contain at least two edges")
    if not np.all(np.diff(z_edges) > 0.0):
        raise ValueError("z_edges_nm must be strictly increasing")

    n_layers = z_edges.size - 1
    thickness = np.diff(z_edges)
    k0 = 2.0 * math.pi / detector.wavelength_nm
    layer_index = np.arange(n_layers, dtype=np.float64)
    alpha_c = np.deg2rad(critical_angle_start_deg + critical_angle_step_deg * layer_index)
    alpha_i = math.radians(detector.alpha_i_deg)

    kz_incident = k0 * np.sqrt(
        (math.sin(alpha_i) ** 2 - alpha_c**2 + 1j * absorption_imag).astype(
            np.complex128
        )
    )
    refl_i_envelope = 1.0 / (1.0 + (alpha_i / np.maximum(alpha_c, 1e-12)) ** 2)
    alpha_incident = reflectivity_scale * refl_i_envelope * np.exp(
        1j * (0.35 + 0.29 * layer_index)
    )
    beta_incident = (
        np.sqrt(np.maximum(0.0, 1.0 - np.abs(alpha_incident) ** 2))
        * np.exp(-beta_loss_per_layer * layer_index)
    ).astype(np.complex128)

    incident_forward = np.empty(n_layers, dtype=np.complex128)
    incident_forward[0] = 1.0 + 0.0j
    for layer in range(1, n_layers):
        incident_forward[layer] = (
            incident_forward[layer - 1]
            * beta_incident[layer - 1]
            * np.exp(1j * kz_incident[layer - 1] * thickness[layer - 1])
        )
    incident_reflected = alpha_incident * incident_forward

    alpha_f = _target_alpha_f(detector)
    alpha_f_rad = np.deg2rad(alpha_f)
    kz_exit = k0 * np.sqrt(
        (
            np.sin(alpha_f_rad)[:, None] ** 2
            - alpha_c[None, :] ** 2
            + 1j * absorption_imag
        ).astype(np.complex128)
    )
    refl_exit_envelope = 1.0 / (
        1.0 + (alpha_f_rad[:, None] / np.maximum(alpha_c[None, :], 1e-12)) ** 2
    )
    alpha_exit = (
        reflectivity_scale
        * refl_exit_envelope
        * np.exp(1j * (0.21 + 0.17 * layer_index[None, :] + 0.03 * alpha_f_rad[:, None]))
    )
    beta_exit = np.sqrt(np.maximum(0.0, 1.0 - np.abs(alpha_exit) ** 2))
    beta_exit = beta_exit * np.exp(-beta_loss_per_layer * layer_index[None, :])

    exit_forward = np.empty_like(kz_exit)
    exit_forward[:, 0] = 1.0 + 0.0j
    for layer in range(1, n_layers):
        exit_forward[:, layer] = (
            exit_forward[:, layer - 1]
            * beta_exit[:, layer - 1]
            * np.exp(1j * kz_exit[:, layer - 1] * thickness[layer - 1])
        )
    exit_reflected = alpha_exit * exit_forward

    return DistortedWaveStack(
        z_edges_nm=np.ascontiguousarray(z_edges),
        kz_incident=np.ascontiguousarray(kz_incident),
        incident_forward=np.ascontiguousarray(incident_forward),
        incident_reflected=np.ascontiguousarray(incident_reflected),
        kz_exit=np.ascontiguousarray(kz_exit),
        exit_forward=np.ascontiguousarray(exit_forward),
        exit_reflected=np.ascontiguousarray(exit_reflected),
        alpha_incident=np.ascontiguousarray(alpha_incident),
        beta_incident=np.ascontiguousarray(beta_incident),
        alpha_exit=np.ascontiguousarray(alpha_exit),
        beta_exit=np.ascontiguousarray(beta_exit),
        incident_angle_deg=float(detector.alpha_i_deg),
        critical_angle_deg=np.rad2deg(alpha_c).astype(np.float64, copy=False),
        target_alpha_f_deg=alpha_f,
    )


def _phase_recurrence(kz: np.ndarray, z: np.ndarray, sign: float) -> np.ndarray:
    kz = np.asarray(kz, dtype=np.complex128).reshape(-1)
    z = np.asarray(z, dtype=np.float64)
    out = np.empty((kz.size, z.size), dtype=np.complex128)
    if z.size == 0:
        return out
    out[:, 0] = np.exp(1j * sign * kz * z[0])
    if z.size == 1:
        return out
    step = np.exp(1j * sign * kz * (z[1] - z[0]))
    for idx in range(1, z.size):
        out[:, idx] = out[:, idx - 1] * step
    return out


def dwba_field_product_grid(stack: DistortedWaveStack, z_centers_nm: np.ndarray) -> np.ndarray:
    """Return conj(exit) * incident on the regular z grid using phase recurrence."""

    z_centers = np.asarray(z_centers_nm, dtype=np.float64)
    n_targets = int(stack.kz_exit.shape[0])
    out = np.empty((n_targets, stack.n_layers, z_centers.size), dtype=np.complex128)
    for layer in range(stack.n_layers):
        local_z = z_centers - stack.z_edges_nm[layer]
        inc_forward = (
            stack.incident_forward[layer]
            * _phase_recurrence(stack.kz_incident[layer : layer + 1], local_z, 1.0)[0]
        )
        inc_reflected = (
            stack.incident_reflected[layer]
            * _phase_recurrence(stack.kz_incident[layer : layer + 1], local_z, -1.0)[0]
        )
        incident = inc_forward + inc_reflected

        exit_forward = (
            stack.exit_forward[:, layer : layer + 1]
            * _phase_recurrence(stack.kz_exit[:, layer], local_z, 1.0)
        )
        exit_reflected = (
            stack.exit_reflected[:, layer : layer + 1]
            * _phase_recurrence(stack.kz_exit[:, layer], local_z, -1.0)
        )
        exit_field = exit_forward + exit_reflected
        out[:, layer, :] = np.conj(exit_field) * incident[None, :]
    return np.ascontiguousarray(out)


def _field_product_for_targets(
    stack: DistortedWaveStack,
    z_nm: np.ndarray,
    layer_ids: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    z = np.asarray(z_nm, dtype=np.float64)
    layer_ids = np.asarray(layer_ids, dtype=np.int64)
    out = np.empty((stop - start, z.size), dtype=np.complex128)
    for layer in range(stack.n_layers):
        mask = layer_ids == layer
        if not np.any(mask):
            continue
        local_z = z[mask] - stack.z_edges_nm[layer]
        incident = (
            stack.incident_forward[layer] * np.exp(1j * stack.kz_incident[layer] * local_z)
            + stack.incident_reflected[layer]
            * np.exp(-1j * stack.kz_incident[layer] * local_z)
        )
        exit_field = (
            stack.exit_forward[start:stop, layer, None]
            * np.exp(1j * stack.kz_exit[start:stop, layer, None] * local_z[None, :])
            + stack.exit_reflected[start:stop, layer, None]
            * np.exp(-1j * stack.kz_exit[start:stop, layer, None] * local_z[None, :])
        )
        out[:, mask] = np.conj(exit_field) * incident[None, :]
    return out


def direct_dwba_atom_amplitude(
    coords: np.ndarray,
    layer_ids: np.ndarray,
    stack: DistortedWaveStack,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    target_chunk: int,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    layer_ids = np.asarray(layer_ids, dtype=np.int64)
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    out = np.empty(qx_flat.size, dtype=np.complex128)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    for start in range(0, qx_flat.size, target_chunk):
        stop = min(start + target_chunk, qx_flat.size)
        phase_xy = x[None, :] * qx_flat[start:stop, None] + y[None, :] * qy_flat[
            start:stop, None
        ]
        field = _field_product_for_targets(stack, z, layer_ids, start, stop)
        out[start:stop] = np.sum(field * np.exp(1j * phase_xy), axis=1)
    return out.reshape(qx.shape)


def finufft_dwba_channel_amplitude(
    coords: np.ndarray,
    layer_ids: np.ndarray,
    stack: DistortedWaveStack,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    eps: float,
    imag_tol: float = 1e-12,
) -> np.ndarray:
    """Evaluate DWBA channel sums with generic FINUFFT type-3 calls.

    The four DWBA channels per layer are expanded as independent Fourier sums.
    This is an exact same-physics baseline only when the effective channel
    ``qz`` values are real. Absorbing or evanescent fields make ``qz`` complex,
    which generic FINUFFT does not support directly.
    """

    try:
        import finufft
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("finufft is not installed") from exc

    coords = np.asarray(coords, dtype=np.float64)
    layer_ids = np.asarray(layer_ids, dtype=np.int64)
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    out = np.zeros(qx_flat.size, dtype=np.complex128)

    incident_channels = (
        (1.0, stack.incident_forward),
        (-1.0, stack.incident_reflected),
    )
    exit_channels = (
        (1.0, stack.exit_forward),
        (-1.0, stack.exit_reflected),
    )
    for layer in range(stack.n_layers):
        mask = layer_ids == layer
        if not np.any(mask):
            continue
        layer_coords = np.ascontiguousarray(coords[mask], dtype=np.float64)
        strengths = np.ones(layer_coords.shape[0], dtype=np.complex128)
        z0 = stack.z_edges_nm[layer]

        for incident_sign, incident_amplitude in incident_channels:
            inc_amp = incident_amplitude[layer]
            for exit_sign, exit_amplitude in exit_channels:
                qz_eff = (
                    incident_sign * stack.kz_incident[layer]
                    - exit_sign * np.conj(stack.kz_exit[:, layer])
                )
                max_imag = float(np.max(np.abs(np.imag(qz_eff))))
                if max_imag > imag_tol:
                    raise ValueError(
                        "FINUFFT DWBA channel baseline requires real effective qz; "
                        f"max imaginary component is {max_imag:.3e}"
                    )
                qz_real = np.ascontiguousarray(np.real(qz_eff), dtype=np.float64)
                coeff = (
                    inc_amp
                    * np.conj(exit_amplitude[:, layer])
                    * np.exp(-1j * qz_eff * z0)
                )
                values = finufft.nufft3d3(
                    np.ascontiguousarray(layer_coords[:, 0]),
                    np.ascontiguousarray(layer_coords[:, 1]),
                    np.ascontiguousarray(layer_coords[:, 2]),
                    strengths,
                    np.ascontiguousarray(qx_flat),
                    np.ascontiguousarray(qy_flat),
                    qz_real,
                    eps=eps,
                    isign=1,
                )
                out += coeff * values
    return out.reshape(qx.shape)


def binned_dwba_direct_amplitude(
    binned,
    field_grid: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    target_chunk: int,
) -> np.ndarray:
    """Direct phase sum over populated cylindrical bins with DWBA field weights."""

    hist = np.asarray(binned.hist)
    active = np.nonzero(hist)
    weights = hist[active].astype(np.complex128, copy=False)
    layer = active[0].astype(np.int64, copy=False)
    z_idx = active[2].astype(np.int64, copy=False)
    r = binned.r_centers[active[1]]
    beta = binned.beta_centers[active[3]]
    x = r * np.cos(beta)
    y = r * np.sin(beta)
    field_flat_index = layer * len(binned.z_centers) + z_idx

    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    out = np.empty(qx_flat.size, dtype=np.complex128)
    field_flat = field_grid.reshape(field_grid.shape[0], -1)
    for start in range(0, qx_flat.size, target_chunk):
        stop = min(start + target_chunk, qx_flat.size)
        phase_xy = x[None, :] * qx_flat[start:stop, None] + y[None, :] * qy_flat[
            start:stop, None
        ]
        field = field_flat[start:stop, :][:, field_flat_index]
        out[start:stop] = np.sum(weights[None, :] * field * np.exp(1j * phase_xy), axis=1)
    return out.reshape(qx.shape)


def build_prepared_dwba_geometry(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    max_mode: int | None,
    enable_mode_pruning: bool = False,
    mode_pruning_margin: int = 32,
    mode_pruning_bin_size: int = 1,
) -> PreparedDwbaGeometry:
    n_phi = int(binned.n_phi)
    requested_indices, requested_modes = _mode_indices(n_phi, max_mode)
    requested_max_mode = (
        int(np.max(np.abs(requested_modes))) if requested_modes.size else 0
    )
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qperp_flat = np.sqrt(qx_flat * qx_flat + qy_flat * qy_flat)
    phi_flat = np.arctan2(qy_flat, qx_flat)
    r_centers = np.asarray(binned.r_centers, dtype=np.float64)

    cutoffs = _build_mode_cutoffs(
        qperp_flat,
        r_centers,
        max_mode=requested_max_mode,
        enable_pruning=enable_mode_pruning,
        margin=mode_pruning_margin,
        cutoff_bin_size=mode_pruning_bin_size,
    )
    active_max_mode = int(np.max(cutoffs)) if cutoffs.size else 0
    indices, modes = _mode_indices(n_phi, active_max_mode)

    mode_phase = np.exp(0.5j * np.pi * modes)
    radial = special.jv(modes[None, None, :], qperp_flat[:, None, None] * r_centers[None, :, None])
    angular = np.exp(1j * phi_flat[:, None] * modes[None, :])
    kernel = radial * mode_phase[None, None, :] * angular[:, None, :]
    if enable_mode_pruning:
        keep = np.abs(modes)[None, None, :] <= cutoffs[:, :, None]
        kernel = np.where(keep, kernel, 0.0)
        mode_work_fraction = float(
            np.count_nonzero(keep) / (cutoffs.size * requested_indices.size)
        )
    else:
        mode_work_fraction = 1.0
    hhat = np.fft.fft(np.asarray(binned.hist), axis=-1)
    hcoef = np.take(hhat, indices, axis=-1) * np.exp(-0.5j * (2.0 * np.pi / n_phi) * modes)
    return PreparedDwbaGeometry(
        indices=np.ascontiguousarray(indices),
        modes=np.ascontiguousarray(modes),
        kernel=np.ascontiguousarray(kernel, dtype=np.complex128),
        n_phi=n_phi,
        hcoef=np.ascontiguousarray(hcoef, dtype=np.complex128),
        cutoffs=np.ascontiguousarray(cutoffs, dtype=np.int64),
        requested_max_mode=requested_max_mode,
        max_mode=active_max_mode,
        mode_pruning=enable_mode_pruning,
        cutoff_margin=int(mode_pruning_margin),
        cutoff_bin_size=int(mode_pruning_bin_size),
        cutoff_min=int(np.min(cutoffs)) if cutoffs.size else 0,
        cutoff_mean=float(np.mean(cutoffs)) if cutoffs.size else 0.0,
        mode_work_fraction=mode_work_fraction,
    )


def execute_prepared_dwba_geometry(
    binned,
    geometry: PreparedDwbaGeometry,
    field_grid: np.ndarray,
    out_shape: tuple[int, int],
    *,
    target_chunk: int,
) -> np.ndarray:
    hcoef = geometry.hcoef
    if hcoef is None:
        delta = 2.0 * np.pi / geometry.n_phi
        hhat = np.fft.fft(np.asarray(binned.hist), axis=-1)
        hcoef = np.take(hhat, geometry.indices, axis=-1) * np.exp(
            -0.5j * delta * geometry.modes
        )

    out = np.empty(field_grid.shape[0], dtype=np.complex128)
    for start in range(0, field_grid.shape[0], target_chunk):
        stop = min(start + target_chunk, field_grid.shape[0])
        source = np.einsum(
            "tlz,lrzh->trh",
            field_grid[start:stop],
            hcoef,
            optimize=True,
        )
        out[start:stop] = np.einsum(
            "trh,trh->t",
            source,
            geometry.kernel[start:stop],
            optimize=True,
        )
    return out.reshape(out_shape)


def build_prepared_dwba_miller_geometry(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    max_mode: int | None,
    extra_order: int,
    enable_mode_pruning: bool,
    mode_pruning_margin: int,
    mode_pruning_bin_size: int,
    complex_dtype: str,
) -> PreparedDwbaMillerGeometry:
    n_phi = int(binned.n_phi)
    requested_max_mode = _positive_max_mode(n_phi, max_mode)
    complex_dtype_np = np.dtype(complex_dtype)
    if complex_dtype_np not in (np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("complex_dtype must be complex64 or complex128")
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qperp_flat = np.sqrt(qx_flat * qx_flat + qy_flat * qy_flat)
    r_centers = np.asarray(binned.r_centers, dtype=np.float64)
    cutoffs = _build_mode_cutoffs(
        qperp_flat,
        r_centers,
        max_mode=requested_max_mode,
        enable_pruning=enable_mode_pruning,
        margin=mode_pruning_margin,
        cutoff_bin_size=mode_pruning_bin_size,
    )
    active_max_mode = int(np.max(cutoffs)) if cutoffs.size else 0
    mode_work_fraction = float(
        np.sum(cutoffs + 1) / (cutoffs.size * (requested_max_mode + 1))
    )
    hist = np.asarray(binned.hist)
    if np.iscomplexobj(hist):
        raise ValueError("DWBA Miller path assumes a real source histogram")
    hhat = np.fft.rfft(hist, axis=-1)[..., : active_max_mode + 1]
    hhat = np.ascontiguousarray(hhat, dtype=complex_dtype_np)
    modes = np.arange(active_max_mode + 1, dtype=np.float64)
    center_phase = np.exp(-0.5j * (2.0 * np.pi / n_phi) * modes).astype(
        complex_dtype_np,
        copy=False,
    )
    hhat *= center_phase
    return PreparedDwbaMillerGeometry(
        qperp=np.ascontiguousarray(qperp_flat, dtype=np.float64),
        phi=np.ascontiguousarray(np.arctan2(qy_flat, qx_flat), dtype=np.float64),
        r_centers=np.ascontiguousarray(r_centers, dtype=np.float64),
        hhat=hhat,
        cutoffs=np.ascontiguousarray(cutoffs, dtype=np.int64),
        n_phi=n_phi,
        requested_max_mode=requested_max_mode,
        max_mode=active_max_mode,
        extra_order=int(extra_order),
        complex_dtype=complex_dtype_np,
        mode_pruning=enable_mode_pruning,
        cutoff_margin=int(mode_pruning_margin),
        cutoff_bin_size=int(mode_pruning_bin_size),
        cutoff_min=int(np.min(cutoffs)) if cutoffs.size else 0,
        cutoff_mean=float(np.mean(cutoffs)) if cutoffs.size else 0.0,
        mode_work_fraction=mode_work_fraction,
    )


def _dwba_channel_terms(
    stack: DistortedWaveStack,
    layer: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    incident_channels = (
        (1.0, stack.incident_forward[layer]),
        (-1.0, stack.incident_reflected[layer]),
    )
    exit_channels = (
        (1.0, stack.exit_forward[:, layer]),
        (-1.0, stack.exit_reflected[:, layer]),
    )
    terms: list[tuple[np.ndarray, np.ndarray]] = []
    for incident_sign, incident_amplitude in incident_channels:
        for exit_sign, exit_amplitude in exit_channels:
            qz_eff = (
                incident_sign * stack.kz_incident[layer]
                - exit_sign * np.conj(stack.kz_exit[:, layer])
            )
            coeff = incident_amplitude * np.conj(exit_amplitude)
            terms.append((np.ascontiguousarray(qz_eff), np.ascontiguousarray(coeff)))
    return tuple(terms)


def _unique_complex_groups(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(values, dtype=np.complex128), return_inverse=True)
    return (
        np.ascontiguousarray(unique, dtype=np.complex128),
        np.ascontiguousarray(inverse.astype(np.int64, copy=False)),
    )


def execute_prepared_dwba_miller_geometry(
    binned,
    geometry: PreparedDwbaMillerGeometry,
    stack: DistortedWaveStack,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Evaluate DWBA channels through the existing C++ Miller contraction."""

    cpp = _cpp_solvers_module()
    suffix = "64" if geometry.complex_dtype == np.dtype("complex64") else ""
    contract_name = f"giwaxs_contract_half_modes_miller_qz_reduced{suffix}"
    if not hasattr(cpp, contract_name):
        raise RuntimeError("DWBA Miller path requires qz-reduced C++ GIWAXS support")
    contract = getattr(cpp, contract_name)

    hhat = geometry.hhat

    n_targets = int(geometry.qperp.size)
    n_layers = int(hhat.shape[0])
    n_z = int(len(binned.z_centers))
    z_centers = np.asarray(binned.z_centers, dtype=np.float64)
    out = np.zeros(n_targets, dtype=geometry.complex_dtype)

    for layer in range(n_layers):
        local_z = z_centers - stack.z_edges_nm[layer]
        for qz_eff, coeff in _dwba_channel_terms(stack, layer):
            qz_unique, qz_group = _unique_complex_groups(qz_eff)
            z_phase_groups = np.exp(1j * qz_unique[:, None] * local_z[None, :])
            z_phase_groups = np.ascontiguousarray(z_phase_groups, dtype=geometry.complex_dtype)
            form_factors = np.zeros((n_layers, n_targets), dtype=geometry.complex_dtype)
            form_factors[layer] = np.asarray(coeff, dtype=geometry.complex_dtype)
            group_cutoffs = _group_cutoffs_by_qz(
                qz_group,
                geometry.cutoffs,
                int(qz_unique.size),
            )
            channel = contract(
                hhat,
                z_phase_groups,
                qz_group,
                geometry.qperp,
                geometry.phi,
                geometry.r_centers,
                np.ascontiguousarray(form_factors),
                geometry.cutoffs,
                group_cutoffs,
                geometry.n_phi,
                geometry.max_mode,
                geometry.extra_order,
            )
            out += np.asarray(channel, dtype=geometry.complex_dtype)
    return np.asarray(out, dtype=np.complex128).reshape(out_shape)


def summarize_stack(stack: DistortedWaveStack) -> dict[str, Any]:
    min_exit = float(np.min(stack.target_alpha_f_deg))
    max_critical = float(np.max(stack.critical_angle_deg))
    return {
        "n_layers": stack.n_layers,
        "z_edges_nm": stack.z_edges_nm,
        "incident_angle_deg": float(stack.incident_angle_deg),
        "target_alpha_f_min_deg": min_exit,
        "critical_angle_range_deg": [
            float(np.min(stack.critical_angle_deg)),
            max_critical,
        ],
        "propagating_incident_above_all_layers": bool(
            stack.incident_angle_deg > max_critical
        ),
        "propagating_exit_above_all_layers": bool(min_exit > max_critical),
        "alpha_incident_abs_range": [
            float(np.min(np.abs(stack.alpha_incident))),
            float(np.max(np.abs(stack.alpha_incident))),
        ],
        "beta_incident_abs_range": [
            float(np.min(np.abs(stack.beta_incident))),
            float(np.max(np.abs(stack.beta_incident))),
        ],
        "alpha_exit_abs_range": [
            float(np.min(np.abs(stack.alpha_exit))),
            float(np.max(np.abs(stack.alpha_exit))),
        ],
        "beta_exit_abs_range": [
            float(np.min(np.abs(stack.beta_exit))),
            float(np.max(np.abs(stack.beta_exit))),
        ],
    }


def run_case(args: argparse.Namespace, detector: GixsDetector, n_atoms: int, case_index: int) -> dict[str, Any]:
    coords, layer_ids, z_edges = make_synthetic_multilayer(
        n_atoms=n_atoms,
        n_layers=args.n_layers,
        radius_nm=args.radius_nm,
        layer_thickness_nm=args.layer_thickness_nm,
        seed=args.seed + case_index,
    )
    stack = build_distorted_wave_stack(
        detector,
        z_edges,
        critical_angle_start_deg=args.critical_angle_start_deg,
        critical_angle_step_deg=args.critical_angle_step_deg,
        absorption_imag=args.absorption_imag,
        reflectivity_scale=args.reflectivity_scale,
        beta_loss_per_layer=args.beta_loss_per_layer,
    )

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            element_indices=layer_ids,
            n_elements=args.n_layers,
            n_r=args.n_r,
            n_z=args.n_z,
            n_phi=args.n_phi,
            r_max=args.radius_nm,
            z_range=(float(z_edges[0]), float(z_edges[-1])),
            backend=args.hist_backend,
            hist_dtype=np.dtype(args.hist_dtype),
        ),
        args.repeats,
    )

    field_grid, field_s, field_times = median_time(
        lambda: dwba_field_product_grid(stack, binned.z_centers),
        args.repeats,
    )
    binned_direct, binned_direct_s, binned_direct_times = median_time(
        lambda: binned_dwba_direct_amplitude(
            binned,
            field_grid,
            detector.qx,
            detector.qy,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )
    geometry, geometry_s, geometry_times = median_time(
        lambda: build_prepared_dwba_geometry(
            binned,
            detector.qx,
            detector.qy,
            max_mode=args.max_mode,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
        ),
        args.repeats,
    )
    prepared, prepared_s, prepared_times = median_time(
        lambda: execute_prepared_dwba_geometry(
            binned,
            geometry,
            field_grid,
            detector.shape,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )

    atom_direct = None
    atom_direct_s = None
    atom_direct_times: list[float] = []
    if n_atoms <= args.direct_atom_limit:
        atom_direct, atom_direct_s, atom_direct_times = median_time(
            lambda: direct_dwba_atom_amplitude(
                coords,
                layer_ids,
                stack,
                detector.qx,
                detector.qy,
                target_chunk=args.target_chunk,
            ),
            max(1, min(args.repeats, 3)),
        )

    prepared_i = intensity(prepared)
    binned_i = intensity(binned_direct)
    atom_i = None if atom_direct is None else intensity(atom_direct)
    active_bins = int(np.count_nonzero(np.asarray(binned.hist)))
    row = {
        "atoms": int(n_atoms),
        "layers": int(args.n_layers),
        "targets": int(detector.qx.size),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "n_phi": int(args.n_phi),
        "max_mode": None if args.max_mode is None else int(args.max_mode),
        "prepared_mode_pruning": bool(geometry.mode_pruning),
        "prepared_requested_max_mode": int(geometry.requested_max_mode),
        "prepared_active_max_mode": int(geometry.max_mode),
        "prepared_cutoff_min": int(geometry.cutoff_min),
        "prepared_cutoff_mean": float(geometry.cutoff_mean),
        "prepared_mode_work_fraction": float(geometry.mode_work_fraction),
        "active_bin_count": active_bins,
        "active_bin_fraction": float(active_bins / np.asarray(binned.hist).size),
        "hist_s": hist_s,
        "field_recurrence_s": field_s,
        "binned_direct_s": binned_direct_s,
        "prepared_geometry_build_s": geometry_s,
        "prepared_hot_s": prepared_s,
        "atom_direct_s": atom_direct_s,
        "prepared_rel_l2_vs_binned_direct": relative_l2(prepared, binned_direct),
        "prepared_intensity_rel_l2_vs_binned_direct": relative_l2(prepared_i, binned_i),
        "binned_direct_rel_l2_vs_atom_direct": None
        if atom_direct is None
        else relative_l2(binned_direct, atom_direct),
        "binned_direct_intensity_rel_l2_vs_atom_direct": None
        if atom_i is None
        else relative_l2(binned_i, atom_i),
        "prepared_speedup_vs_binned_direct": float(binned_direct_s / prepared_s)
        if prepared_s
        else None,
        "atom_direct_speedup_vs_prepared": None
        if atom_direct_s is None or prepared_s == 0.0
        else float(atom_direct_s / prepared_s),
        "times": {
            "hist": hist_times,
            "field_recurrence": field_times,
            "binned_direct": binned_direct_times,
            "prepared_geometry_build": geometry_times,
            "prepared_hot": prepared_times,
            "atom_direct": atom_direct_times,
        },
    }
    print(
        "{atoms}: field={field:.4f}s binned={binned_s:.4f}s prepared={prepared_s:.4f}s "
        "mode_work={mode_work:.1%} prep_int_l2={l2:.3e} atom_l2={atom_l2}".format(
            atoms=n_atoms,
            field=field_s,
            binned_s=binned_direct_s,
            prepared_s=prepared_s,
            mode_work=float(row["prepared_mode_work_fraction"]),
            l2=float(row["prepared_intensity_rel_l2_vs_binned_direct"]),
            atom_l2="n/a"
            if row["binned_direct_intensity_rel_l2_vs_atom_direct"] is None
            else f"{float(row['binned_direct_intensity_rel_l2_vs_atom_direct']):.3e}",
        )
    )
    return row | {"stack": summarize_stack(stack)}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_sci(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3e}"


def _fmt_speed(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}x"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# GIWAXS DWBA Multilayer Recurrence Prototype",
        "",
        "This benchmark keeps the electromagnetic side deliberately narrow: layer-wise",
        "distorted incident and exit fields are supplied by an alpha/beta recurrence,",
        "then the scattering contraction is evaluated with the existing cylindrical",
        "harmonic source representation.",
        "",
        "It is a scattering-operator prototype, not a validated Fresnel/Parratt",
        "multilayer simulator.",
        "",
        "## Detector",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    for key, value in summary["detector"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| atoms | layers | grid | targets | active bins | dense mode work | field recurrence s | binned direct s | prepared hot s | prepared/direct | prepared intensity L2 | binned-vs-atom intensity L2 |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        grid = f"{row['n_r']}x{row['n_z']}x{row['n_phi']}"
        lines.append(
            "| {atoms} | {layers} | `{grid}` | {targets} | {active_bins} | "
            "{mode_work} | `{field}` | `{binned}` | `{prepared}` | {speed} | {prep_l2} | {atom_l2} |".format(
                atoms=row["atoms"],
                layers=row["layers"],
                grid=grid,
                targets=row["targets"],
                active_bins=row["active_bin_count"],
                mode_work=_fmt_pct(row["prepared_mode_work_fraction"]),
                field=_fmt(row["field_recurrence_s"]),
                binned=_fmt(row["binned_direct_s"]),
                prepared=_fmt(row["prepared_hot_s"]),
                speed=_fmt_speed(row["prepared_speedup_vs_binned_direct"]),
                prep_l2=_fmt_sci(row["prepared_intensity_rel_l2_vs_binned_direct"]),
                atom_l2=_fmt_sci(row["binned_direct_intensity_rel_l2_vs_atom_direct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `prepared intensity L2` compares the harmonic contraction against direct summation over populated cylindrical bins with the same DWBA field grid.",
            "- `binned-vs-atom intensity L2` is finite binning error against exact atom positions and direct field exponentials.",
            "- The recurrence uses complex reflection amplitudes `alpha` and complex-capable transmission factors `beta`; the default beta values are real attenuating factors.",
            "- The next production step is replacing the synthetic alpha/beta recurrence with a Fresnel/Parratt field provider while preserving the same scattering API.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{summary['config']['out']}`",
            f"- Markdown: `{summary['config']['summary_md']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype DWBA-style multilayer GIWAXS scattering recurrence."
    )
    parser.add_argument("--atoms", type=int, nargs="+", default=[8000, 50000])
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--radius-nm", type=float, default=4.0)
    parser.add_argument("--layer-thickness-nm", type=float, default=2.0)
    parser.add_argument("--n-r", type=int, default=32)
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-phi", type=int, default=192)
    parser.add_argument("--max-mode", type=int, default=95)
    parser.add_argument("--mode-pruning-margin", type=int, default=32)
    parser.add_argument("--mode-pruning-bin-size", type=int, default=1)
    parser.add_argument("--disable-mode-pruning", action="store_true")
    parser.add_argument("--wavelength-nm", type=float, default=0.15406)
    parser.add_argument("--alpha-i-deg", type=float, default=0.2)
    parser.add_argument("--alpha-f-min-deg", type=float, default=0.3)
    parser.add_argument("--alpha-f-max-deg", type=float, default=6.0)
    parser.add_argument("--n-alpha-f", type=int, default=8)
    parser.add_argument("--two-theta-min-deg", type=float, default=-6.0)
    parser.add_argument("--two-theta-max-deg", type=float, default=6.0)
    parser.add_argument("--n-two-theta", type=int, default=12)
    parser.add_argument("--critical-angle-start-deg", type=float, default=0.13)
    parser.add_argument("--critical-angle-step-deg", type=float, default=0.018)
    parser.add_argument("--absorption-imag", type=float, default=2.5e-4)
    parser.add_argument("--reflectivity-scale", type=float, default=0.28)
    parser.add_argument("--beta-loss-per-layer", type=float, default=0.018)
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="numpy")
    parser.add_argument("--hist-dtype", default="float32")
    parser.add_argument("--target-chunk", type=int, default=96)
    parser.add_argument("--direct-atom-limit", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_dwba_multilayer_benchmark.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_dwba_multilayer_benchmark.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = make_giwaxs_detector(
        wavelength_nm=args.wavelength_nm,
        alpha_i_deg=args.alpha_i_deg,
        alpha_f_min_deg=args.alpha_f_min_deg,
        alpha_f_max_deg=args.alpha_f_max_deg,
        n_alpha_f=args.n_alpha_f,
        two_theta_min_deg=args.two_theta_min_deg,
        two_theta_max_deg=args.two_theta_max_deg,
        n_two_theta=args.n_two_theta,
    )
    rows = [run_case(args, detector, atoms, index) for index, atoms in enumerate(args.atoms)]
    summary = {
        "config": _as_jsonable(vars(args) | {"out": args.out, "summary_md": args.summary_md}),
        "detector": summarize_detector(detector),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_as_jsonable(summary), indent=2), encoding="utf-8")
    write_summary(args.summary_md, summary)
    print(args.out)
    print(args.summary_md)


if __name__ == "__main__":
    main()
