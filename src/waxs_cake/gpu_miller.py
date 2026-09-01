"""GPU construction of cylindrical Bessel kernels by Miller recurrence."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


_CUDA_SOURCE = r"""
extern "C" __global__
void miller_kernel64(
    const double* q_perp,
    const double* r_centers,
    float2* out,
    const long long n_q,
    const long long n_r,
    const int n_phi,
    const int max_cutoff,
    const int extra_order
) {
    const long long item = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long n_items = n_q * n_r;
    if (item >= n_items) return;

    const long long iq = item / n_r;
    const long long ir = item - iq * n_r;
    const double x = q_perp[iq] * r_centers[ir];
    const int n_h = max_cutoff + 1;
    float2* row = out + item * n_h;
    for (int h = 0; h <= max_cutoff; ++h) row[h] = make_float2(0.0f, 0.0f);

    if (fabs(x) < 1.0e-300) {
        row[0] = make_float2((float)n_phi, 0.0f);
        return;
    }

    // The publication path never approaches this guard (H=48), but keeping
    // a fixed local array makes register/local-memory use inspectable.
    if (max_cutoff > 255) return;
    double saved[256];
    for (int h = 0; h <= max_cutoff; ++h) saved[h] = 0.0;

    int m = (int)ceil(fabs(x)) + extra_order;
    const int minimum_m = max_cutoff + extra_order;
    if (m < minimum_m) m = minimum_m;
    if (m < max_cutoff) m = max_cutoff;

    double b_next = 0.0;
    double b_curr = 1.0;
    if (m <= max_cutoff) saved[m] = b_curr;
    double even_tail = (m >= 2 && (m & 1) == 0) ? 2.0 : 0.0;

    for (int n = m; n > 0; --n) {
        double b_prev = (2.0 * (double)n / x) * b_curr - b_next;
        const int k = n - 1;
        if (k <= max_cutoff) saved[k] = b_prev;
        if (k >= 2 && (k & 1) == 0) even_tail += 2.0 * b_prev;
        b_next = b_curr;
        b_curr = b_prev;

        if (fabs(b_curr) > 1.0e100 || fabs(b_next) > 1.0e100) {
            b_curr *= 1.0e-100;
            b_next *= 1.0e-100;
            even_tail *= 1.0e-100;
            for (int h = 0; h <= max_cutoff; ++h) saved[h] *= 1.0e-100;
        }
    }

    const double denom = saved[0] + even_tail;
    if (denom == 0.0 || !isfinite(denom)) return;
    const double scale = (double)n_phi / denom;
    for (int h = 0; h <= max_cutoff; ++h) {
        const float value = (float)(saved[h] * scale);
        const int phase = h & 3;
        if (phase == 0) row[h] = make_float2(value, 0.0f);
        else if (phase == 1) row[h] = make_float2(0.0f, value);
        else if (phase == 2) row[h] = make_float2(-value, 0.0f);
        else row[h] = make_float2(0.0f, -value);
    }
}
"""

_RAW_KERNEL: Any | None = None


def _raw_kernel() -> Any:
    global _RAW_KERNEL
    if _RAW_KERNEL is None:
        import cupy as cp

        _RAW_KERNEL = cp.RawKernel(
            _CUDA_SOURCE,
            "miller_kernel64",
            options=("--std=c++11",),
        )
    return _RAW_KERNEL


def warm_gpu_miller_kernel() -> float:
    """Compile/load the CuPy kernel and return cold software-startup time."""

    import cupy as cp

    start = perf_counter()
    _raw_kernel().compile()
    cp.cuda.runtime.deviceSynchronize()
    return perf_counter() - start


def gpu_miller_kernel64(
    q_perp: np.ndarray,
    r_centers: np.ndarray,
    *,
    n_phi: int,
    max_cutoff: int,
    extra_order: int = 64,
    torch: Any,
) -> Any:
    """Return a GPU-resident complex64 ``(q,R,h)`` Miller kernel."""

    import cupy as cp

    q = cp.asarray(np.ascontiguousarray(q_perp, dtype=np.float64))
    r = cp.asarray(np.ascontiguousarray(r_centers, dtype=np.float64))
    if q.ndim != 1 or r.ndim != 1:
        raise ValueError("q_perp and r_centers must be one-dimensional")
    if max_cutoff < 0 or max_cutoff >= int(n_phi) // 2:
        raise ValueError("max_cutoff must satisfy 0 <= H < n_phi / 2")
    if max_cutoff > 255:
        raise ValueError("GPU Miller prototype currently requires H <= 255")
    if extra_order < 0:
        raise ValueError("extra_order must be non-negative")

    out = cp.empty((q.size, r.size, max_cutoff + 1), dtype=cp.complex64)
    threads = 128
    blocks = (q.size * r.size + threads - 1) // threads
    _raw_kernel()(
        (blocks,),
        (threads,),
        (
            q,
            r,
            out,
            np.int64(q.size),
            np.int64(r.size),
            np.int32(n_phi),
            np.int32(max_cutoff),
            np.int32(extra_order),
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    return torch.from_dlpack(out)
