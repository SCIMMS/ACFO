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

extern "C" __global__
void miller_kernel128(
    const double* q_perp,
    const double* r_centers,
    double2* out,
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
    double2* row = out + item * n_h;
    for (int h = 0; h <= max_cutoff; ++h) row[h] = make_double2(0.0, 0.0);

    if (fabs(x) < 1.0e-300) {
        row[0] = make_double2((double)n_phi, 0.0);
        return;
    }
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
        const double value = saved[h] * scale;
        const int phase = h & 3;
        if (phase == 0) row[h] = make_double2(value, 0.0);
        else if (phase == 1) row[h] = make_double2(0.0, value);
        else if (phase == 2) row[h] = make_double2(-value, 0.0);
        else row[h] = make_double2(0.0, -value);
    }
}
"""

_RAW_KERNEL: dict[str, Any] = {}


def _raw_kernel(name: str = "miller_kernel64") -> Any:
    if name not in {"miller_kernel64", "miller_kernel128"}:
        raise ValueError("unknown GPU Miller kernel")
    if name not in _RAW_KERNEL:
        import cupy as cp

        _RAW_KERNEL[name] = cp.RawKernel(
            _CUDA_SOURCE,
            name,
            options=("--std=c++11",),
        )
    return _RAW_KERNEL[name]


def warm_gpu_miller_kernel() -> float:
    """Compile/load the CuPy kernel and return cold software-startup time."""

    import cupy as cp

    start = perf_counter()
    _raw_kernel("miller_kernel64").compile()
    _raw_kernel("miller_kernel128").compile()
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


def gpu_miller_kernel128(
    q_perp: np.ndarray,
    r_centers: np.ndarray,
    *,
    n_phi: int,
    max_cutoff: int,
    extra_order: int = 64,
    torch: Any,
) -> Any:
    """Return a GPU-resident complex128 ``(q,R,h)`` Miller kernel."""

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

    out = cp.empty((q.size, r.size, max_cutoff + 1), dtype=cp.complex128)
    threads = 128
    blocks = (q.size * r.size + threads - 1) // threads
    _raw_kernel("miller_kernel128")(
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


def _gpu_miller_kernel_torch_resident(
    q_perp: Any,
    r_centers: Any,
    *,
    n_phi: int,
    max_cutoff: int,
    extra_order: int,
    torch: Any,
    kernel_name: str,
    complex_dtype: Any,
) -> Any:
    """Build a Miller block into Torch-owned memory on Torch's CUDA stream.

    The historical helpers above accept NumPy inputs and synchronize while
    transferring ownership of a CuPy allocation through DLPack.  Repeating
    that path for every q block would add host transfers, allocator retention,
    and cross-stream lifetime hazards.  This resident path keeps q, radius,
    output and all downstream contractions in the Torch allocator and orders
    the raw kernel on the current Torch stream.
    """

    import cupy as cp

    if not q_perp.is_cuda or not r_centers.is_cuda:
        raise ValueError("resident GPU Miller inputs must be CUDA tensors")
    if q_perp.ndim != 1 or r_centers.ndim != 1:
        raise ValueError("q_perp and r_centers must be one-dimensional")
    if q_perp.dtype != torch.float64 or r_centers.dtype != torch.float64:
        raise ValueError("resident GPU Miller inputs must use float64")
    if q_perp.device != r_centers.device:
        raise ValueError("q_perp and r_centers must share one CUDA device")
    if max_cutoff < 0 or max_cutoff >= int(n_phi) // 2:
        raise ValueError("max_cutoff must satisfy 0 <= H < n_phi / 2")
    if max_cutoff > 255:
        raise ValueError("GPU Miller prototype currently requires H <= 255")
    if extra_order < 0:
        raise ValueError("extra_order must be non-negative")

    q_perp = q_perp.contiguous()
    r_centers = r_centers.contiguous()
    out = torch.empty(
        (q_perp.numel(), r_centers.numel(), max_cutoff + 1),
        dtype=complex_dtype,
        device=q_perp.device,
    )
    q_view = cp.from_dlpack(q_perp)
    r_view = cp.from_dlpack(r_centers)
    out_view = cp.from_dlpack(out)
    threads = 128
    blocks = (q_perp.numel() * r_centers.numel() + threads - 1) // threads
    torch_stream = torch.cuda.current_stream(q_perp.device)
    cupy_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    _raw_kernel(kernel_name)(
        (blocks,),
        (threads,),
        (
            q_view,
            r_view,
            out_view,
            np.int64(q_perp.numel()),
            np.int64(r_centers.numel()),
            np.int32(n_phi),
            np.int32(max_cutoff),
            np.int32(extra_order),
        ),
        stream=cupy_stream,
    )
    return out


def gpu_miller_kernel64_torch_resident(
    q_perp: Any,
    r_centers: Any,
    *,
    n_phi: int,
    max_cutoff: int,
    extra_order: int = 64,
    torch: Any,
) -> Any:
    """Return a Torch-owned complex64 Miller block without host transfers."""

    return _gpu_miller_kernel_torch_resident(
        q_perp,
        r_centers,
        n_phi=n_phi,
        max_cutoff=max_cutoff,
        extra_order=extra_order,
        torch=torch,
        kernel_name="miller_kernel64",
        complex_dtype=torch.complex64,
    )


def gpu_miller_kernel128_torch_resident(
    q_perp: Any,
    r_centers: Any,
    *,
    n_phi: int,
    max_cutoff: int,
    extra_order: int = 64,
    torch: Any,
) -> Any:
    """Return a Torch-owned complex128 Miller block without host transfers."""

    return _gpu_miller_kernel_torch_resident(
        q_perp,
        r_centers,
        n_phi=n_phi,
        max_cutoff=max_cutoff,
        extra_order=extra_order,
        torch=torch,
        kernel_name="miller_kernel128",
        complex_dtype=torch.complex128,
    )
