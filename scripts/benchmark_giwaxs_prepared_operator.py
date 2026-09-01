from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import make_cylindrical_histogram, water_box_side_nm  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


@dataclass(frozen=True)
class GixsDetector:
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    alpha_f_deg: np.ndarray
    two_theta_deg: np.ndarray
    wavelength_nm: float
    alpha_i_deg: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.qx.shape

    @property
    def qmag(self) -> np.ndarray:
        return np.sqrt(self.qx * self.qx + self.qy * self.qy + self.qz * self.qz)

    @property
    def qperp(self) -> np.ndarray:
        return np.sqrt(self.qx * self.qx + self.qy * self.qy)

    @property
    def phi(self) -> np.ndarray:
        return np.arctan2(self.qy, self.qx)


@dataclass(frozen=True)
class PreparedGiwaxsGeometry:
    indices: np.ndarray
    modes: np.ndarray
    z_phase: np.ndarray
    kernel: np.ndarray
    n_phi: int


@dataclass(frozen=True)
class PreparedGiwaxsMillerGeometry:
    qperp: np.ndarray
    phi: np.ndarray
    z_phase: np.ndarray
    z_phase_groups: np.ndarray
    qz_group: np.ndarray
    r_centers: np.ndarray
    form_factors: np.ndarray
    cutoffs: np.ndarray
    group_cutoffs: np.ndarray
    kernel_pos: np.ndarray | None
    kernel_neg: np.ndarray | None
    n_phi: int
    requested_max_mode: int
    max_mode: int
    extra_order: int
    qz_reduction: bool
    qz_group_count: int
    kernel_precompute: bool
    kernel_memory_mb: float
    complex_dtype: np.dtype
    mode_pruning: bool
    cutoff_margin: int
    cutoff_bin_size: int
    cutoff_min: int
    cutoff_mean: float
    mode_work_fraction: float


@dataclass(frozen=True)
class SparseProfileHalfModes:
    profile_e: np.ndarray
    profile_r: np.ndarray
    profile_z: np.ndarray
    active_hhat: np.ndarray
    r_profile_starts: np.ndarray
    r_profile_counts: np.ndarray
    active_profile_count: int
    active_profile_fraction: float
    active_beta_count: int
    active_beta_fraction: float


def synthetic_box(n_atoms: int, side_nm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5 * side_nm, 0.5 * side_nm, size=(n_atoms, 3))


def make_giwaxs_detector(
    *,
    wavelength_nm: float,
    alpha_i_deg: float,
    alpha_f_min_deg: float,
    alpha_f_max_deg: float,
    n_alpha_f: int,
    two_theta_min_deg: float,
    two_theta_max_deg: float,
    n_two_theta: int,
) -> GixsDetector:
    """Return a kinematic GIWAXS-style reciprocal-space detector map.

    Coordinates follow a common grazing-incidence convention:

    qx = k cos(alpha_f) sin(2theta_f)
    qy = k [cos(alpha_f) cos(2theta_f) - cos(alpha_i)]
    qz = k [sin(alpha_f) + sin(alpha_i)]

    This is a support experiment for fixed detector geometry, not a full
    DWBA/refraction GIWAXS simulator.
    """

    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")
    if n_alpha_f <= 0 or n_two_theta <= 0:
        raise ValueError("detector dimensions must be positive")

    alpha_f_deg = np.linspace(alpha_f_min_deg, alpha_f_max_deg, n_alpha_f)
    two_theta_deg = np.linspace(two_theta_min_deg, two_theta_max_deg, n_two_theta)
    alpha_f = np.deg2rad(alpha_f_deg)[:, None]
    two_theta = np.deg2rad(two_theta_deg)[None, :]
    alpha_i = math.radians(alpha_i_deg)
    k = 2.0 * math.pi / wavelength_nm

    qx = k * np.cos(alpha_f) * np.sin(two_theta)
    qy = k * (np.cos(alpha_f) * np.cos(two_theta) - math.cos(alpha_i))
    qz = k * (np.sin(alpha_f) + math.sin(alpha_i)) * np.ones_like(qx)
    return GixsDetector(
        qx=qx,
        qy=qy,
        qz=qz,
        alpha_f_deg=alpha_f_deg,
        two_theta_deg=two_theta_deg,
        wavelength_nm=wavelength_nm,
        alpha_i_deg=alpha_i_deg,
    )


def median_time(func, repeats: int) -> tuple[Any, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def direct_atom_amplitude(
    coords: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    target_chunk: int,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qz_flat = np.ravel(qz).astype(np.float64)
    out = np.empty(qx_flat.size, dtype=np.complex128)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    for start in range(0, qx_flat.size, target_chunk):
        stop = min(start + target_chunk, qx_flat.size)
        phase = (
            x[:, None] * qx_flat[None, start:stop]
            + y[:, None] * qy_flat[None, start:stop]
            + z[:, None] * qz_flat[None, start:stop]
        )
        out[start:stop] = np.sum(np.exp(1j * phase), axis=0)
    return out.reshape(qx.shape)


def nufft_arbitrary_amplitude(
    coords: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    import finufft

    coords = np.asarray(coords, dtype=np.float64)
    values = finufft.nufft3d3(
        np.ascontiguousarray(coords[:, 0]),
        np.ascontiguousarray(coords[:, 1]),
        np.ascontiguousarray(coords[:, 2]),
        np.ones(coords.shape[0], dtype=np.complex128),
        np.ascontiguousarray(np.ravel(qx).astype(np.float64)),
        np.ascontiguousarray(np.ravel(qy).astype(np.float64)),
        np.ascontiguousarray(np.ravel(qz).astype(np.float64)),
        eps=eps,
        isign=1,
    )
    return values.reshape(qx.shape)


def _mode_indices(n_phi: int, max_mode: int | None) -> tuple[np.ndarray, np.ndarray]:
    indices_all = np.arange(n_phi, dtype=np.int64)
    signed = indices_all.copy()
    signed[signed >= (n_phi + 1) // 2] -= n_phi
    if max_mode is not None:
        max_mode = int(max_mode)
        if max_mode < 0:
            raise ValueError("max_mode must be non-negative")
        keep = np.abs(signed) <= max_mode
        indices = indices_all[keep]
        signed = signed[keep]
    else:
        indices = indices_all
    order = np.argsort(signed)
    return indices[order], signed[order]


def _positive_max_mode(n_phi: int, max_mode: int | None) -> int:
    if max_mode is None:
        max_mode = n_phi // 2 - 1
    max_mode = int(max_mode)
    if max_mode < 0:
        raise ValueError("max_mode must be non-negative")
    if max_mode >= n_phi // 2:
        raise ValueError("Miller half-spectrum path requires max_mode < n_phi / 2")
    return max_mode


def _cpp_solvers_module():
    try:
        from waxs_cake import _cpp_solvers
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Build waxs_cake._cpp_solvers before using Miller GIWAXS") from exc
    if not hasattr(_cpp_solvers, "giwaxs_contract_half_modes_miller"):
        raise RuntimeError("waxs_cake._cpp_solvers lacks giwaxs_contract_half_modes_miller")
    return _cpp_solvers


def _build_mode_cutoffs(
    qperp: np.ndarray,
    r_centers: np.ndarray,
    *,
    max_mode: int,
    enable_pruning: bool,
    margin: int,
    cutoff_bin_size: int,
) -> np.ndarray:
    if margin < 0:
        raise ValueError("mode pruning margin must be non-negative")
    if cutoff_bin_size <= 0:
        raise ValueError("mode pruning bin size must be positive")
    if not enable_pruning:
        return np.full((qperp.size, r_centers.size), max_mode, dtype=np.int64)

    cutoffs = np.ceil(np.abs(qperp[:, None]) * r_centers[None, :] + margin).astype(
        np.int64,
        copy=False,
    )
    np.clip(cutoffs, 0, max_mode, out=cutoffs)
    if cutoff_bin_size > 1:
        cutoffs = ((cutoffs + cutoff_bin_size - 1) // cutoff_bin_size) * cutoff_bin_size
        np.clip(cutoffs, 0, max_mode, out=cutoffs)
    return np.ascontiguousarray(cutoffs, dtype=np.int64)


def _group_cutoffs_by_qz(qz_group: np.ndarray, cutoffs: np.ndarray, n_groups: int) -> np.ndarray:
    group_cutoffs = np.zeros((n_groups, cutoffs.shape[1]), dtype=np.int64)
    np.maximum.at(group_cutoffs, qz_group, cutoffs)
    return np.ascontiguousarray(group_cutoffs, dtype=np.int64)


def build_sparse_profile_half_modes(
    binned,
    *,
    max_mode: int,
    n_phi: int,
    complex_dtype: np.dtype,
) -> SparseProfileHalfModes:
    hist = np.asarray(binned.hist)
    if np.iscomplexobj(hist):
        raise ValueError("Sparse GIWAXS path assumes a real histogram")
    profile_mask = np.any(hist != 0, axis=-1)
    active = np.nonzero(profile_mask)
    n_profiles = int(active[0].size)
    n_elements, n_r, n_z, _ = hist.shape
    if n_profiles == 0:
        hhat = np.empty((0, max_mode + 1), dtype=complex_dtype)
        empty = np.empty(0, dtype=np.int64)
        return SparseProfileHalfModes(
            profile_e=empty,
            profile_r=empty,
            profile_z=empty,
            active_hhat=hhat,
            r_profile_starts=np.zeros(n_r, dtype=np.int64),
            r_profile_counts=np.zeros(n_r, dtype=np.int64),
            active_profile_count=0,
            active_profile_fraction=0.0,
            active_beta_count=0,
            active_beta_fraction=0.0,
        )

    profile_e = active[0].astype(np.int64, copy=False)
    profile_r = active[1].astype(np.int64, copy=False)
    profile_z = active[2].astype(np.int64, copy=False)
    order = np.argsort(profile_r, kind="stable")
    profile_e = np.ascontiguousarray(profile_e[order], dtype=np.int64)
    profile_r = np.ascontiguousarray(profile_r[order], dtype=np.int64)
    profile_z = np.ascontiguousarray(profile_z[order], dtype=np.int64)
    active_profiles = np.ascontiguousarray(hist[active][order])
    hhat = np.fft.rfft(active_profiles, axis=-1)[..., : max_mode + 1]
    hhat = np.ascontiguousarray(hhat, dtype=complex_dtype)
    modes = np.arange(max_mode + 1, dtype=np.float64)
    center_phase = np.exp(-0.5j * (2.0 * np.pi / n_phi) * modes).astype(
        complex_dtype,
        copy=False,
    )
    hhat *= center_phase

    counts = np.bincount(profile_r, minlength=n_r).astype(np.int64, copy=False)
    starts = np.empty(n_r, dtype=np.int64)
    if n_r:
        starts[0] = 0
        if n_r > 1:
            starts[1:] = np.cumsum(counts[:-1])
    active_beta_count = int(np.count_nonzero(hist))
    return SparseProfileHalfModes(
        profile_e=profile_e,
        profile_r=profile_r,
        profile_z=profile_z,
        active_hhat=hhat,
        r_profile_starts=np.ascontiguousarray(starts, dtype=np.int64),
        r_profile_counts=np.ascontiguousarray(counts, dtype=np.int64),
        active_profile_count=n_profiles,
        active_profile_fraction=float(n_profiles / (n_elements * n_r * n_z)),
        active_beta_count=active_beta_count,
        active_beta_fraction=float(active_beta_count / hist.size),
    )


def prepared_giwaxs_amplitude(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    max_mode: int | None,
    target_chunk: int,
) -> np.ndarray:
    """Evaluate a fixed GIWAXS q-map from cylindrical source harmonics."""

    n_phi = binned.n_phi
    indices, modes = _mode_indices(n_phi, max_mode)
    delta = 2.0 * np.pi / n_phi
    hhat = np.fft.fft(binned.hist, axis=-1)
    hcoef = np.take(hhat, indices, axis=-1) * np.exp(-0.5j * delta * modes)

    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qz_flat = np.ravel(qz).astype(np.float64)
    qperp_flat = np.sqrt(qx_flat * qx_flat + qy_flat * qy_flat)
    phi_flat = np.arctan2(qy_flat, qx_flat)
    out = np.empty(qx_flat.size, dtype=np.complex128)
    r_centers = np.asarray(binned.r_centers, dtype=np.float64)
    z_centers = np.asarray(binned.z_centers, dtype=np.float64)
    mode_phase = np.exp(0.5j * np.pi * modes)

    for start in range(0, qx_flat.size, target_chunk):
        stop = min(start + target_chunk, qx_flat.size)
        qz_block = qz_flat[start:stop]
        qperp_block = qperp_flat[start:stop]
        phi_block = phi_flat[start:stop]
        z_phase = np.exp(1j * qz_block[:, None] * z_centers[None, :])
        source = np.einsum("tz,erzh->terh", z_phase, hcoef, optimize=True)
        source = np.sum(source, axis=1)
        x = qperp_block[:, None, None] * r_centers[None, :, None]
        radial = special.jv(modes[None, None, :], x)
        angular = np.exp(1j * phi_block[:, None] * modes[None, :])
        kernel = radial * mode_phase[None, None, :] * angular[:, None, :]
        out[start:stop] = np.einsum("trh,trh->t", source, kernel, optimize=True)
    return out.reshape(qx.shape)


def build_prepared_giwaxs_geometry(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    max_mode: int | None,
) -> PreparedGiwaxsGeometry:
    n_phi = binned.n_phi
    indices, modes = _mode_indices(n_phi, max_mode)
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qz_flat = np.ravel(qz).astype(np.float64)
    qperp_flat = np.sqrt(qx_flat * qx_flat + qy_flat * qy_flat)
    phi_flat = np.arctan2(qy_flat, qx_flat)
    r_centers = np.asarray(binned.r_centers, dtype=np.float64)
    z_centers = np.asarray(binned.z_centers, dtype=np.float64)
    mode_phase = np.exp(0.5j * np.pi * modes)
    z_phase = np.exp(1j * qz_flat[:, None] * z_centers[None, :])
    x = qperp_flat[:, None, None] * r_centers[None, :, None]
    radial = special.jv(modes[None, None, :], x)
    angular = np.exp(1j * phi_flat[:, None] * modes[None, :])
    kernel = radial * mode_phase[None, None, :] * angular[:, None, :]
    return PreparedGiwaxsGeometry(
        indices=indices,
        modes=modes,
        z_phase=np.ascontiguousarray(z_phase, dtype=np.complex128),
        kernel=np.ascontiguousarray(kernel, dtype=np.complex128),
        n_phi=n_phi,
    )


def execute_prepared_giwaxs_geometry(
    binned,
    geometry: PreparedGiwaxsGeometry,
    out_shape: tuple[int, int],
) -> np.ndarray:
    delta = 2.0 * np.pi / geometry.n_phi
    hhat = np.fft.fft(binned.hist, axis=-1)
    hcoef = np.take(hhat, geometry.indices, axis=-1) * np.exp(
        -0.5j * delta * geometry.modes
    )
    source = np.einsum("tz,erzh->terh", geometry.z_phase, hcoef, optimize=True)
    source = np.sum(source, axis=1)
    out = np.einsum("trh,trh->t", source, geometry.kernel, optimize=True)
    return out.reshape(out_shape)


def build_prepared_giwaxs_miller_geometry(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    max_mode: int | None,
    extra_order: int,
    enable_mode_pruning: bool,
    mode_pruning_margin: int,
    mode_pruning_bin_size: int,
    enable_qz_reduction: bool,
    precompute_kernel: bool,
    complex_dtype: str,
) -> PreparedGiwaxsMillerGeometry:
    n_phi = int(binned.n_phi)
    requested_max_mode = _positive_max_mode(n_phi, max_mode)
    complex_dtype_np = np.dtype(complex_dtype)
    if complex_dtype_np not in (np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("complex_dtype must be complex64 or complex128")
    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qz_flat = np.ravel(qz).astype(np.float64)
    qperp_flat = np.sqrt(qx_flat * qx_flat + qy_flat * qy_flat)
    phi_flat = np.arctan2(qy_flat, qx_flat)
    r_centers = np.asarray(binned.r_centers, dtype=np.float64)
    z_centers = np.asarray(binned.z_centers, dtype=np.float64)
    z_phase = np.exp(1j * qz_flat[:, None] * z_centers[None, :])
    qz_unique, qz_group = np.unique(qz_flat, return_inverse=True)
    z_phase_groups = np.exp(1j * qz_unique[:, None] * z_centers[None, :])
    n_elements = int(np.asarray(binned.hist).shape[0])
    form_factors = np.ones((n_elements, qx_flat.size), dtype=np.complex128)
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
    qz_group = np.ascontiguousarray(qz_group.astype(np.int64, copy=False))
    group_cutoffs = _group_cutoffs_by_qz(qz_group, cutoffs, qz_unique.size)
    kernel_pos = None
    kernel_neg = None
    kernel_memory_mb = 0.0
    if precompute_kernel:
        if not enable_qz_reduction:
            raise ValueError("precompute_kernel requires qz reduction")
        cpp = _cpp_solvers_module()
        kernel_builder = (
            cpp.analytic_kernel_hat_modes_miller64
            if complex_dtype_np == np.dtype("complex64")
            else cpp.analytic_kernel_hat_modes_miller
        )
        khat = kernel_builder(
            np.ascontiguousarray(qperp_flat, dtype=np.float64),
            np.ascontiguousarray(r_centers, dtype=np.float64),
            n_phi,
            active_max_mode,
            int(extra_order),
        )
        khat = np.asarray(khat, dtype=complex_dtype_np) / float(n_phi)
        modes = np.arange(active_max_mode + 1, dtype=np.float64)
        angular = np.exp(1j * phi_flat[:, None] * modes[None, :])
        kernel_pos = np.ascontiguousarray(khat * angular[:, None, :], dtype=complex_dtype_np)
        kernel_neg = np.ascontiguousarray(
            khat * np.conj(angular[:, None, :]),
            dtype=complex_dtype_np,
        )
        kernel_memory_mb = float(
            (kernel_pos.nbytes + kernel_neg.nbytes) / (1024.0 * 1024.0)
        )
    return PreparedGiwaxsMillerGeometry(
        qperp=np.ascontiguousarray(qperp_flat, dtype=np.float64),
        phi=np.ascontiguousarray(phi_flat, dtype=np.float64),
        z_phase=np.ascontiguousarray(z_phase, dtype=complex_dtype_np),
        z_phase_groups=np.ascontiguousarray(z_phase_groups, dtype=complex_dtype_np),
        qz_group=qz_group,
        r_centers=np.ascontiguousarray(r_centers, dtype=np.float64),
        form_factors=np.ascontiguousarray(form_factors, dtype=complex_dtype_np),
        cutoffs=np.ascontiguousarray(cutoffs, dtype=np.int64),
        group_cutoffs=group_cutoffs,
        kernel_pos=kernel_pos,
        kernel_neg=kernel_neg,
        n_phi=n_phi,
        requested_max_mode=requested_max_mode,
        max_mode=active_max_mode,
        extra_order=int(extra_order),
        qz_reduction=enable_qz_reduction,
        qz_group_count=int(qz_unique.size),
        kernel_precompute=precompute_kernel,
        kernel_memory_mb=kernel_memory_mb,
        complex_dtype=complex_dtype_np,
        mode_pruning=enable_mode_pruning,
        cutoff_margin=int(mode_pruning_margin),
        cutoff_bin_size=int(mode_pruning_bin_size),
        cutoff_min=int(np.min(cutoffs)) if cutoffs.size else 0,
        cutoff_mean=float(np.mean(cutoffs)) if cutoffs.size else 0.0,
        mode_work_fraction=mode_work_fraction,
    )


def execute_prepared_giwaxs_miller_geometry(
    binned,
    geometry: PreparedGiwaxsMillerGeometry,
    out_shape: tuple[int, int],
    *,
    source_backend: str,
) -> np.ndarray:
    if source_backend not in {"dense", "sparse"}:
        raise ValueError("source_backend must be 'dense' or 'sparse'")
    cpp = _cpp_solvers_module()
    suffix = "64" if geometry.complex_dtype == np.dtype("complex64") else ""
    if source_backend == "sparse":
        contract_name = f"giwaxs_contract_sparse_profiles_miller_qz_reduced{suffix}"
        if not geometry.qz_reduction or not hasattr(cpp, contract_name):
            raise RuntimeError("Sparse GIWAXS source backend requires qz-reduced C++ support")
        sparse = build_sparse_profile_half_modes(
            binned,
            max_mode=geometry.max_mode,
            n_phi=geometry.n_phi,
            complex_dtype=geometry.complex_dtype,
        )
        contract = getattr(cpp, contract_name)
        out = contract(
            sparse.profile_e,
            sparse.profile_r,
            sparse.profile_z,
            sparse.active_hhat,
            sparse.r_profile_starts,
            sparse.r_profile_counts,
            geometry.z_phase_groups,
            geometry.qz_group,
            geometry.qperp,
            geometry.phi,
            geometry.r_centers,
            geometry.form_factors,
            geometry.cutoffs,
            geometry.n_phi,
            geometry.max_mode,
            geometry.extra_order,
        )
        return np.asarray(out).reshape(out_shape)

    hist = np.asarray(binned.hist)
    if np.iscomplexobj(hist):
        raise ValueError("Miller half-spectrum GIWAXS path assumes a real histogram")
    hhat = np.fft.rfft(hist, axis=-1)[..., : geometry.max_mode + 1]
    hhat = np.ascontiguousarray(hhat, dtype=geometry.complex_dtype)
    modes = np.arange(geometry.max_mode + 1, dtype=np.float64)
    center_phase = np.exp(-0.5j * (2.0 * np.pi / geometry.n_phi) * modes).astype(
        geometry.complex_dtype,
        copy=False,
    )
    hhat *= center_phase
    if (
        geometry.kernel_pos is not None
        and geometry.kernel_neg is not None
        and hasattr(cpp, f"giwaxs_contract_half_modes_kernel_qz_reduced{suffix}")
    ):
        contract = getattr(cpp, f"giwaxs_contract_half_modes_kernel_qz_reduced{suffix}")
        out = contract(
            hhat,
            geometry.z_phase_groups,
            geometry.qz_group,
            geometry.kernel_pos,
            geometry.kernel_neg,
            geometry.form_factors,
            geometry.cutoffs,
            geometry.group_cutoffs,
            geometry.max_mode,
        )
    elif geometry.qz_reduction and hasattr(
        cpp, f"giwaxs_contract_half_modes_miller_qz_reduced{suffix}"
    ):
        contract = getattr(cpp, f"giwaxs_contract_half_modes_miller_qz_reduced{suffix}")
        out = contract(
            hhat,
            geometry.z_phase_groups,
            geometry.qz_group,
            geometry.qperp,
            geometry.phi,
            geometry.r_centers,
            geometry.form_factors,
            geometry.cutoffs,
            geometry.group_cutoffs,
            geometry.n_phi,
            geometry.max_mode,
            geometry.extra_order,
        )
    else:
        contract = getattr(cpp, f"giwaxs_contract_half_modes_miller{suffix}")
        out = contract(
            hhat,
            geometry.z_phase,
            geometry.qperp,
            geometry.phi,
            geometry.r_centers,
            geometry.form_factors,
            geometry.cutoffs,
            geometry.n_phi,
            geometry.max_mode,
            geometry.extra_order,
        )
    return np.asarray(out).reshape(out_shape)


def binned_direct_amplitude(
    binned,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    target_chunk: int,
) -> np.ndarray:
    """Direct phase sum over populated cylindrical bins."""

    hist = np.asarray(binned.hist)
    active = np.nonzero(hist)
    weights = hist[active].astype(np.complex128, copy=False)
    r = binned.r_centers[active[1]]
    z = binned.z_centers[active[2]]
    beta = binned.beta_centers[active[3]]
    x = r * np.cos(beta)
    y = r * np.sin(beta)

    qx_flat = np.ravel(qx).astype(np.float64)
    qy_flat = np.ravel(qy).astype(np.float64)
    qz_flat = np.ravel(qz).astype(np.float64)
    out = np.empty(qx_flat.size, dtype=np.complex128)
    for start in range(0, qx_flat.size, target_chunk):
        stop = min(start + target_chunk, qx_flat.size)
        phase = (
            x[:, None] * qx_flat[None, start:stop]
            + y[:, None] * qy_flat[None, start:stop]
            + z[:, None] * qz_flat[None, start:stop]
        )
        out[start:stop] = np.sum(weights[:, None] * np.exp(1j * phase), axis=0)
    return out.reshape(qx.shape)


def summarize_detector(detector: GixsDetector) -> dict[str, Any]:
    qmag = detector.qmag
    qperp = detector.qperp
    return {
        "shape": list(detector.shape),
        "targets": int(qmag.size),
        "wavelength_nm": detector.wavelength_nm,
        "alpha_i_deg": detector.alpha_i_deg,
        "alpha_f_range_deg": [
            float(detector.alpha_f_deg[0]),
            float(detector.alpha_f_deg[-1]),
        ],
        "two_theta_range_deg": [
            float(detector.two_theta_deg[0]),
            float(detector.two_theta_deg[-1]),
        ],
        "qmag_min_inv_nm": float(np.min(qmag)),
        "qmag_max_inv_nm": float(np.max(qmag)),
        "qperp_max_inv_nm": float(np.max(qperp)),
        "qz_min_inv_nm": float(np.min(detector.qz)),
        "qz_max_inv_nm": float(np.max(detector.qz)),
    }


def _fmt_float(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_sci(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3e}"


def _fmt_speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _source_stats(binned) -> dict[str, float | int]:
    hist = np.asarray(binned.hist)
    active_profiles = int(np.count_nonzero(np.any(hist != 0, axis=-1)))
    active_beta = int(np.count_nonzero(hist))
    n_elements, n_r, n_z, _ = hist.shape
    return {
        "active_profile_count": active_profiles,
        "active_profile_fraction": float(active_profiles / (n_elements * n_r * n_z)),
        "active_beta_count": active_beta,
        "active_beta_fraction": float(active_beta / hist.size),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# GIWAXS Prepared-Operator Support Benchmark",
        "",
        "This is a kinematic GIWAXS detector-map support experiment. It is not a",
        "full GIWAXS DWBA/refraction simulation. The goal is to show that the",
        "same cylindrical harmonic source representation used for WAXS cake maps",
        "can evaluate a fixed grazing-incidence X-ray detector q-map.",
        "",
        "## Detector",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    detector = summary["detector"]
    for key, value in detector.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| atoms | bins | targets | backend | source | dtype | qz groups | active profiles | kernel MB | mode cutoff | mode work | binned direct s | dense build s | dense hot s | Miller build s | Miller hot s | FINUFFT s | Miller intensity L2 | Miller speedup vs binned direct | FINUFFT / Miller |",
            "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {atoms} | {bins} | {targets} | {backend} | {source} | {dtype} | "
            "{qz_groups} | {active_profiles} | {kernel_mb:.1f} | "
            "{mode_cutoff} | {mode_work} | {binned_direct_s:.4f} | "
            "{dense_build_s} | {dense_hot_s} | {miller_build_s} | {miller_hot_s} | "
            "{nufft_s} | {miller_int_rel} | {miller_speedup} | {nufft_miller_speedup} |".format(
                atoms=row["atoms"],
                bins=f"{row['n_r']}x{row['n_z']}x{row['n_phi']}",
                targets=row["targets"],
                backend=row.get("miller_backend", "miller_target_fused"),
                source=row.get("source_backend", "dense"),
                dtype=row.get("miller_complex_dtype", "complex128"),
                qz_groups=row.get("miller_qz_group_count", 0),
                active_profiles=_fmt_pct(row.get("active_profile_fraction")),
                kernel_mb=float(row.get("miller_kernel_memory_mb", 0.0)),
                mode_cutoff=(
                    f"{row['miller_cutoff_min']}/"
                    f"{row['miller_cutoff_mean']:.1f}/"
                    f"{row['miller_max_mode']}"
                ),
                mode_work=_fmt_pct(row["miller_mode_work_fraction"]),
                binned_direct_s=row["binned_direct_s"],
                dense_build_s=_fmt_float(row["prepared_geometry_build_s"]),
                dense_hot_s=_fmt_float(row["prepared_hot_s"]),
                miller_build_s=_fmt_float(row["prepared_miller_geometry_build_s"]),
                miller_hot_s=_fmt_float(row["prepared_miller_hot_s"]),
                nufft_s=_fmt_float(row["nufft_s"]),
                miller_int_rel=_fmt_sci(
                    row["prepared_miller_hot_intensity_rel_l2_vs_binned_direct"]
                ),
                miller_speedup=_fmt_speedup(
                    row["prepared_miller_hot_speedup_vs_binned_direct"]
                ),
                nufft_miller_speedup=_fmt_speedup(
                    row["nufft_speedup_vs_prepared_miller_hot"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Claim-Support Readout",
            "",
            "- GIWAXS extends the X-ray side from ordinary transmission/cake-map WAXS",
            "  to a fixed grazing-incidence detector q-map without changing the",
            "  source representation idea.",
            "- The support claim should stay kinematic: this benchmark does not claim",
            "  full GIWAXS sample-surface physics, refraction, multiple scattering,",
            "  or DWBA correction.",
            "- The useful patent language is a prepared operator for fixed X-ray",
            "  detector geometries, including detector cake maps and",
            "  grazing-incidence detector q-maps.",
            "- The optimized GIWAXS path uses a fused half-spectrum Miller recurrence",
            "  contraction, avoiding the dense SciPy Bessel kernel materialization",
            "  used by the validation path.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(args, detector: GixsDetector, n_atoms: int, case_index: int) -> dict[str, Any]:
    side_nm = args.box_side_nm or water_box_side_nm(n_atoms)
    coords = synthetic_box(n_atoms, side_nm, args.seed + case_index)
    r_max = math.sqrt(0.5) * side_nm
    z_range = (-0.5 * side_nm, 0.5 * side_nm)

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            n_r=args.n_r,
            n_z=args.n_z,
            n_phi=args.n_phi,
            r_max=r_max,
            z_range=z_range,
            backend=args.hist_backend,
            hist_dtype=np.dtype(args.hist_dtype),
        ),
        args.repeats,
    )
    source_stats = _source_stats(binned)

    qx, qy, qz = detector.qx, detector.qy, detector.qz
    binned_direct, binned_direct_s, binned_direct_times = median_time(
        lambda: binned_direct_amplitude(
            binned,
            qx,
            qy,
            qz,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )

    prepared = None
    prepared_s = None
    prepared_times: list[float] = []
    geometry_build_s = None
    geometry_build_times: list[float] = []
    prepared_hot = None
    prepared_hot_s = None
    prepared_hot_times: list[float] = []
    if not args.skip_dense_prepared:
        prepared, prepared_s, prepared_times = median_time(
            lambda: prepared_giwaxs_amplitude(
                binned,
                qx,
                qy,
                qz,
                max_mode=args.max_mode,
                target_chunk=args.target_chunk,
            ),
            args.repeats,
        )
        geometry, geometry_build_s, geometry_build_times = median_time(
            lambda: build_prepared_giwaxs_geometry(
                binned,
                qx,
                qy,
                qz,
                max_mode=args.max_mode,
            ),
            args.repeats,
        )
        prepared_hot, prepared_hot_s, prepared_hot_times = median_time(
            lambda: execute_prepared_giwaxs_geometry(
                binned,
                geometry,
                qx.shape,
            ),
            args.repeats,
        )

    miller_geometry, miller_geometry_build_s, miller_geometry_build_times = median_time(
        lambda: build_prepared_giwaxs_miller_geometry(
            binned,
            qx,
            qy,
            qz,
            max_mode=args.max_mode,
            extra_order=args.miller_extra_order,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
            enable_qz_reduction=not args.disable_qz_reduction,
            precompute_kernel=args.precompute_pruned_kernel,
            complex_dtype=args.miller_complex_dtype,
        ),
        args.repeats,
    )
    prepared_miller_hot, prepared_miller_hot_s, prepared_miller_hot_times = median_time(
        lambda: execute_prepared_giwaxs_miller_geometry(
            binned,
            miller_geometry,
            qx.shape,
            source_backend=args.source_backend,
        ),
        args.repeats,
    )

    atom_direct = None
    atom_direct_s = None
    atom_direct_times: list[float] = []
    if n_atoms <= args.direct_atom_limit:
        atom_direct, atom_direct_s, atom_direct_times = median_time(
            lambda: direct_atom_amplitude(
                coords,
                qx,
                qy,
                qz,
                target_chunk=args.target_chunk,
            ),
            max(1, min(args.repeats, 3)),
        )

    nufft = None
    nufft_s = None
    nufft_times: list[float] = []
    if not args.skip_nufft:
        try:
            nufft, nufft_s, nufft_times = median_time(
                lambda: nufft_arbitrary_amplitude(
                    coords,
                    qx,
                    qy,
                    qz,
                    eps=args.nufft_eps,
                ),
                max(1, min(args.repeats, 3)),
            )
        except RuntimeError as exc:
            print(f"FINUFFT skipped for {n_atoms}: {exc}")

    row = {
        "atoms": int(n_atoms),
        "box_side_nm": float(side_nm),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "n_phi": int(args.n_phi),
        "max_mode": None if args.max_mode is None else int(args.max_mode),
        "targets": int(qx.size),
        "source_backend": args.source_backend,
        "active_profile_count": source_stats["active_profile_count"],
        "active_profile_fraction": source_stats["active_profile_fraction"],
        "active_beta_count": source_stats["active_beta_count"],
        "active_beta_fraction": source_stats["active_beta_fraction"],
        "hist_s": hist_s,
        "binned_direct_s": binned_direct_s,
        "prepared_s": prepared_s,
        "prepared_geometry_build_s": geometry_build_s,
        "prepared_hot_s": prepared_hot_s,
        "prepared_miller_geometry_build_s": miller_geometry_build_s,
        "prepared_miller_hot_s": prepared_miller_hot_s,
        "miller_requested_max_mode": int(miller_geometry.requested_max_mode),
        "miller_max_mode": int(miller_geometry.max_mode),
        "miller_backend": "kernel_qz_reduced"
        if miller_geometry.kernel_precompute
        else "miller_qz_reduced"
        if miller_geometry.qz_reduction
        else "miller_target_fused",
        "miller_qz_reduction": bool(miller_geometry.qz_reduction),
        "miller_qz_group_count": int(miller_geometry.qz_group_count),
        "miller_kernel_precompute": bool(miller_geometry.kernel_precompute),
        "miller_kernel_memory_mb": float(miller_geometry.kernel_memory_mb),
        "miller_complex_dtype": str(miller_geometry.complex_dtype),
        "miller_mode_pruning": bool(miller_geometry.mode_pruning),
        "miller_cutoff_margin": int(miller_geometry.cutoff_margin),
        "miller_cutoff_bin_size": int(miller_geometry.cutoff_bin_size),
        "miller_cutoff_min": int(miller_geometry.cutoff_min),
        "miller_cutoff_mean": float(miller_geometry.cutoff_mean),
        "miller_mode_work_fraction": float(miller_geometry.mode_work_fraction),
        "atom_direct_s": atom_direct_s,
        "nufft_s": nufft_s,
        "prepared_rel_l2_vs_binned_direct": None
        if prepared is None
        else relative_l2(prepared, binned_direct),
        "prepared_intensity_rel_l2_vs_binned_direct": None
        if prepared is None
        else relative_l2(
            intensity(prepared),
            intensity(binned_direct),
        ),
        "prepared_hot_rel_l2_vs_binned_direct": None
        if prepared_hot is None
        else relative_l2(
            prepared_hot,
            binned_direct,
        ),
        "prepared_hot_intensity_rel_l2_vs_binned_direct": None
        if prepared_hot is None
        else relative_l2(
            intensity(prepared_hot),
            intensity(binned_direct),
        ),
        "prepared_miller_hot_rel_l2_vs_binned_direct": relative_l2(
            prepared_miller_hot,
            binned_direct,
        ),
        "prepared_miller_hot_intensity_rel_l2_vs_binned_direct": relative_l2(
            intensity(prepared_miller_hot),
            intensity(binned_direct),
        ),
        "prepared_hot_rel_l2_vs_prepared": None
        if prepared_hot is None or prepared is None
        else relative_l2(prepared_hot, prepared),
        "prepared_miller_hot_rel_l2_vs_prepared_hot": None
        if prepared_hot is None
        else relative_l2(prepared_miller_hot, prepared_hot),
        "atom_direct_rel_l2_vs_prepared": None
        if atom_direct is None or prepared is None
        else relative_l2(atom_direct, prepared),
        "atom_direct_intensity_rel_l2_vs_prepared": None
        if atom_direct is None or prepared is None
        else relative_l2(intensity(atom_direct), intensity(prepared)),
        "nufft_rel_l2_vs_atom_direct": None
        if nufft is None or atom_direct is None
        else relative_l2(nufft, atom_direct),
        "nufft_intensity_rel_l2_vs_atom_direct": None
        if nufft is None or atom_direct is None
        else relative_l2(intensity(nufft), intensity(atom_direct)),
        "prepared_speedup_vs_binned_direct": binned_direct_s / prepared_s
        if prepared_s is not None and prepared_s
        else None,
        "prepared_hot_speedup_vs_binned_direct": binned_direct_s / prepared_hot_s
        if prepared_hot_s is not None and prepared_hot_s
        else None,
        "prepared_miller_hot_speedup_vs_binned_direct": binned_direct_s
        / prepared_miller_hot_s
        if prepared_miller_hot_s
        else None,
        "atom_direct_speedup_vs_prepared": None
        if atom_direct_s is None or prepared_s is None
        else atom_direct_s / prepared_s,
        "atom_direct_speedup_vs_prepared_hot": None
        if atom_direct_s is None or prepared_hot_s is None
        else atom_direct_s / prepared_hot_s,
        "atom_direct_speedup_vs_prepared_miller_hot": None
        if atom_direct_s is None
        else atom_direct_s / prepared_miller_hot_s,
        "nufft_speedup_vs_prepared": None
        if nufft_s is None or prepared_s is None
        else nufft_s / prepared_s,
        "nufft_speedup_vs_prepared_hot": None
        if nufft_s is None or prepared_hot_s is None
        else nufft_s / prepared_hot_s,
        "nufft_speedup_vs_prepared_miller_hot": None
        if nufft_s is None
        else nufft_s / prepared_miller_hot_s,
        "times": {
            "hist": hist_times,
            "binned_direct": binned_direct_times,
            "prepared": prepared_times,
            "prepared_geometry_build": geometry_build_times,
            "prepared_hot": prepared_hot_times,
            "prepared_miller_geometry_build": miller_geometry_build_times,
            "prepared_miller_hot": prepared_miller_hot_times,
            "atom_direct": atom_direct_times,
            "nufft": nufft_times,
        },
    }
    print(
        f"{n_atoms}: hist={hist_s:.4f}s binned_direct={binned_direct_s:.4f}s "
        f"prepared={_fmt_float(prepared_s)}s hot={_fmt_float(prepared_hot_s)}s "
        f"miller_hot={prepared_miller_hot_s:.4f}s nufft={nufft_s} "
        f"backend={row['miller_backend']} qz_groups={miller_geometry.qz_group_count} "
        f"source={args.source_backend} active_profiles={100.0 * source_stats['active_profile_fraction']:.1f}% "
        f"dtype={miller_geometry.complex_dtype} "
        f"cutoff_mean={miller_geometry.cutoff_mean:.1f}/{miller_geometry.requested_max_mode} "
        f"mode_work={100.0 * miller_geometry.mode_work_fraction:.1f}% "
        f"miller_int_l2={row['prepared_miller_hot_intensity_rel_l2_vs_binned_direct']:.3e}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a prepared cylindrical-harmonic operator on a kinematic GIWAXS q-map."
    )
    parser.add_argument("--atoms", type=int, nargs="+", default=[10000, 50000])
    parser.add_argument("--box-side-nm", type=float, default=None)
    parser.add_argument("--n-r", type=int, default=36)
    parser.add_argument("--n-z", type=int, default=36)
    parser.add_argument("--n-phi", type=int, default=320)
    parser.add_argument("--max-mode", type=int, default=None)
    parser.add_argument("--wavelength-nm", type=float, default=0.15406)
    parser.add_argument("--alpha-i-deg", type=float, default=0.2)
    parser.add_argument("--alpha-f-min-deg", type=float, default=0.3)
    parser.add_argument("--alpha-f-max-deg", type=float, default=18.0)
    parser.add_argument("--n-alpha-f", type=int, default=24)
    parser.add_argument("--two-theta-min-deg", type=float, default=-18.0)
    parser.add_argument("--two-theta-max-deg", type=float, default=18.0)
    parser.add_argument("--n-two-theta", type=int, default=48)
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument("--hist-dtype", default="float32")
    parser.add_argument("--target-chunk", type=int, default=96)
    parser.add_argument("--miller-extra-order", type=int, default=64)
    parser.add_argument("--source-backend", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--mode-pruning-margin", type=int, default=32)
    parser.add_argument("--mode-pruning-bin-size", type=int, default=1)
    parser.add_argument("--disable-mode-pruning", action="store_true")
    parser.add_argument("--disable-qz-reduction", action="store_true")
    parser.add_argument("--precompute-pruned-kernel", action="store_true")
    parser.add_argument(
        "--miller-complex-dtype",
        choices=["complex64", "complex128"],
        default="complex128",
    )
    parser.add_argument("--skip-dense-prepared", action="store_true")
    parser.add_argument("--direct-atom-limit", type=int, default=20000)
    parser.add_argument("--skip-nufft", action="store_true")
    parser.add_argument("--nufft-eps", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_prepared_operator_benchmark.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_prepared_operator_benchmark.md",
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
    rows = [run_case(args, detector, n_atoms, i) for i, n_atoms in enumerate(args.atoms)]
    summary = {
        "config": vars(args) | {"out": str(args.out), "summary_md": str(args.summary_md)},
        "detector": summarize_detector(detector),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(args.summary_md, summary)
    print(args.out)
    print(args.summary_md)


if __name__ == "__main__":
    main()
