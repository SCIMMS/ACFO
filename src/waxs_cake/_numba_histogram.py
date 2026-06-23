"""Optional numba histogram kernels."""

from __future__ import annotations

import numpy as np

try:
    from numba import get_num_threads, get_thread_id, njit, prange
except ImportError:  # pragma: no cover - exercised when numba is not installed
    njit = None


if njit is not None:

    @njit(cache=True, fastmath=True)
    def _histogram_single_unweighted_kernel(
        coords: np.ndarray,
        n_r: int,
        n_z: int,
        n_phi: int,
        r_max: float,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        hist = np.zeros(n_r * n_z * n_phi, dtype=np.int64)
        r_scale = n_r / r_max
        z_scale = n_z / (z_max - z_min)
        phi_scale = n_phi / (2.0 * np.pi)

        for atom_idx in range(coords.shape[0]):
            x = coords[atom_idx, 0]
            y = coords[atom_idx, 1]
            z = coords[atom_idx, 2]

            radius = np.sqrt(x * x + y * y)
            r_idx = int(radius * r_scale)
            if r_idx < 0:
                r_idx = 0
            elif r_idx >= n_r:
                r_idx = n_r - 1

            z_idx = int((z - z_min) * z_scale)
            if z_idx < 0:
                z_idx = 0
            elif z_idx >= n_z:
                z_idx = n_z - 1

            beta = np.arctan2(y, x)
            if beta < 0.0:
                beta += 2.0 * np.pi
            beta_idx = int(beta * phi_scale)
            if beta_idx < 0:
                beta_idx = 0
            elif beta_idx >= n_phi:
                beta_idx = n_phi - 1

            flat_idx = (r_idx * n_z + z_idx) * n_phi + beta_idx
            hist[flat_idx] += 1

        return hist


    @njit(fastmath=True, parallel=True)
    def _histogram_single_unweighted_parallel_kernel(
        coords: np.ndarray,
        n_r: int,
        n_z: int,
        n_phi: int,
        r_max: float,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        n_bins = n_r * n_z * n_phi
        n_threads = get_num_threads()
        local = np.zeros((n_threads, n_bins), dtype=np.int64)
        r_scale = n_r / r_max
        z_scale = n_z / (z_max - z_min)
        phi_scale = n_phi / (2.0 * np.pi)

        for atom_idx in prange(coords.shape[0]):
            thread_id = get_thread_id()
            x = coords[atom_idx, 0]
            y = coords[atom_idx, 1]
            z = coords[atom_idx, 2]

            radius = np.sqrt(x * x + y * y)
            r_idx = int(radius * r_scale)
            if r_idx < 0:
                r_idx = 0
            elif r_idx >= n_r:
                r_idx = n_r - 1

            z_idx = int((z - z_min) * z_scale)
            if z_idx < 0:
                z_idx = 0
            elif z_idx >= n_z:
                z_idx = n_z - 1

            beta = np.arctan2(y, x)
            if beta < 0.0:
                beta += 2.0 * np.pi
            beta_idx = int(beta * phi_scale)
            if beta_idx < 0:
                beta_idx = 0
            elif beta_idx >= n_phi:
                beta_idx = n_phi - 1

            flat_idx = (r_idx * n_z + z_idx) * n_phi + beta_idx
            local[thread_id, flat_idx] += 1

        hist = np.zeros(n_bins, dtype=np.int64)
        for thread_idx in range(n_threads):
            for bin_idx in range(n_bins):
                hist[bin_idx] += local[thread_idx, bin_idx]
        return hist


def histogram_single_unweighted(
    coords: np.ndarray,
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
    r_max: float,
    z_min: float,
    z_max: float,
    parallel: bool = False,
) -> np.ndarray:
    """Return a flat int64 histogram for one unweighted element."""

    if njit is None:
        raise RuntimeError("numba is not installed")
    kernel = (
        _histogram_single_unweighted_parallel_kernel
        if parallel
        else _histogram_single_unweighted_kernel
    )
    return kernel(
        np.ascontiguousarray(coords),
        int(n_r),
        int(n_z),
        int(n_phi),
        float(r_max),
        float(z_min),
        float(z_max),
    )
