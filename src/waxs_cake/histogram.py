"""Cylindrical binning for ring-wise WAXS cake-map solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_TWO_PI = 2.0 * np.pi
_SUPPORTED_HIST_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in ("int64", "uint32", "float32", "float64", "complex64", "complex128")
)


def _normalize_hist_dtype(dtype: np.dtype | str | None) -> np.dtype | None:
    if dtype is None:
        return None
    normalized = np.dtype(dtype)
    if normalized not in _SUPPORTED_HIST_DTYPES:
        raise ValueError(
            "hist_dtype must be one of int64, uint32, float32, float64, "
            "complex64, or complex128"
        )
    return normalized


def _regular_axes(
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
    r_max: float,
    z_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_r <= 0 or n_z <= 0 or n_phi <= 0:
        raise ValueError("n_r, n_z, and n_phi must be positive")
    r_max = float(r_max)
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    z_min, z_max = map(float, z_range)
    if z_min >= z_max:
        raise ValueError("z_range must be increasing")

    r_edges = np.linspace(0.0, r_max, n_r + 1)
    z_edges = np.linspace(z_min, z_max, n_z + 1)
    beta_edges = np.linspace(0.0, _TWO_PI, n_phi + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    return r_edges, z_edges, beta_edges, r_centers, z_centers, beta_centers


def _validate_weight_dtype(
    weights: np.ndarray | None,
    hist_dtype: np.dtype | None,
) -> None:
    if weights is None:
        if hist_dtype is not None and hist_dtype.kind == "c":
            raise ValueError("complex hist_dtype requires atom_weights")
        return

    has_imaginary_weights = np.iscomplexobj(weights) and bool(np.any(weights.imag))
    if has_imaginary_weights:
        if hist_dtype is not None and hist_dtype.kind != "c":
            raise ValueError("complex atom_weights require complex hist_dtype")
    elif hist_dtype is not None and hist_dtype.kind not in {"f", "c"}:
        raise ValueError("weighted histograms require float or complex hist_dtype")


def encode_elements(
    elements: Iterable[str],
    *,
    element_order: Iterable[str] | None = None,
    dtype: np.dtype | str = np.int64,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode per-atom element labels as small integer IDs.

    This is the preferred setup step for multi-element production runs: call it
    once after loading a structure, then pass the returned indices to
    ``make_cylindrical_histogram_indexed``.
    """

    out_dtype = np.dtype(dtype)
    if not np.issubdtype(out_dtype, np.integer):
        raise ValueError("dtype must be an integer dtype")

    if element_order is not None:
        ordered_elements = tuple(str(e) for e in element_order)
        if not ordered_elements:
            raise ValueError("at least one element is required")
        if not isinstance(elements, np.ndarray):
            mapping = {element: i for i, element in enumerate(ordered_elements)}
            try:
                count = len(elements)  # type: ignore[arg-type]
            except TypeError:
                count = -1
            try:
                element_indices = np.fromiter(
                    (mapping[str(e)] for e in elements),
                    dtype=out_dtype,
                    count=count,
                )
            except KeyError as exc:
                missing = str(exc.args[0])
                raise ValueError(
                    f"element {missing!r} is missing from element_order"
                ) from None
            return np.ascontiguousarray(element_indices), ordered_elements

    atom_elements = np.asarray(elements)
    if atom_elements.ndim == 0:
        atom_elements = np.asarray(list(elements))
    if atom_elements.ndim != 1:
        raise ValueError("elements must be one-dimensional")

    atom_element_keys = (
        atom_elements
        if atom_elements.dtype.kind in {"U", "O"}
        else atom_elements.astype(str, copy=False)
    )
    if element_order is None:
        ordered_elements = tuple(dict.fromkeys(str(e) for e in atom_element_keys))
    else:
        ordered_elements = tuple(str(e) for e in element_order)
    if not ordered_elements:
        raise ValueError("at least one element is required")

    element_indices = np.empty(atom_element_keys.shape[0], dtype=out_dtype)
    matched = np.zeros(atom_element_keys.shape[0], dtype=bool)
    for i, element in enumerate(ordered_elements):
        mask = atom_element_keys == element
        element_indices[mask] = i
        matched |= mask
    if not np.all(matched):
        missing = str(atom_element_keys[~matched][0])
        raise ValueError(f"element {missing!r} is missing from element_order")

    return np.ascontiguousarray(element_indices), ordered_elements


def _histogram_flat_indices(
    flat_idx: np.ndarray,
    *,
    n_bins: int,
    atom_weights: np.ndarray | None,
    hist_dtype: np.dtype | None,
    backend: str,
    validate_indices: bool = True,
) -> np.ndarray:
    if backend not in {"numpy", "cpp"}:
        raise ValueError("backend must be 'numpy' or 'cpp'")
    flat_idx = np.ascontiguousarray(flat_idx, dtype=np.int64)
    if flat_idx.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if validate_indices and backend == "numpy" and flat_idx.size and (
        flat_idx.min(initial=0) < 0 or flat_idx.max(initial=0) >= n_bins
    ):
        raise ValueError("flat_indices contains an out-of-range bin")

    weights = None if atom_weights is None else np.asarray(atom_weights)
    if weights is not None and weights.shape != flat_idx.shape:
        raise ValueError("atom_weights must have one entry per index")
    _validate_weight_dtype(weights, hist_dtype)

    if backend == "cpp":
        try:
            from . import _cpp_histogram
        except ImportError as exc:
            raise ImportError(
                "backend='cpp' requires the pybind11 extension to be built. "
                "Install build dependencies and run `python setup.py build_ext --inplace` "
                "or `python -m pip install -e .`."
            ) from exc

        if weights is None:
            if hist_dtype == np.dtype("uint32"):
                return _cpp_histogram.histogram_flat_unweighted_uint32(
                    flat_idx,
                    n_bins,
                    validate_indices,
                )
            if hist_dtype == np.dtype("float32"):
                return _cpp_histogram.histogram_flat_unweighted_float32(
                    flat_idx,
                    n_bins,
                    validate_indices,
                )
            hist = _cpp_histogram.histogram_flat_unweighted(
                flat_idx,
                n_bins,
                validate_indices,
            )
            return hist.astype(hist_dtype, copy=False) if hist_dtype is not None else hist

        if np.iscomplexobj(weights) and np.any(weights.imag):
            if hist_dtype == np.dtype("complex64"):
                return _cpp_histogram.histogram_flat_weighted_complex64(
                    flat_idx,
                    np.ascontiguousarray(weights, dtype=np.complex64),
                    n_bins,
                    validate_indices,
                )
            hist = _cpp_histogram.histogram_flat_weighted_complex(
                flat_idx,
                np.ascontiguousarray(weights, dtype=np.complex128),
                n_bins,
                validate_indices,
            )
            return hist.astype(hist_dtype, copy=False) if hist_dtype is not None else hist

        if hist_dtype == np.dtype("float32"):
            return _cpp_histogram.histogram_flat_weighted_real_float32(
                flat_idx,
                np.ascontiguousarray(weights.real, dtype=np.float32),
                n_bins,
                validate_indices,
            )
        hist = _cpp_histogram.histogram_flat_weighted_real(
            flat_idx,
            np.ascontiguousarray(weights.real, dtype=np.float64),
            n_bins,
            validate_indices,
        )
        return hist.astype(hist_dtype, copy=False) if hist_dtype is not None else hist

    if weights is None:
        hist = np.bincount(flat_idx, minlength=n_bins)
    else:
        weights = np.asarray(weights, dtype=np.complex128)
        hist_real = np.bincount(flat_idx, weights=weights.real, minlength=n_bins)
        if np.any(weights.imag):
            hist_imag = np.bincount(flat_idx, weights=weights.imag, minlength=n_bins)
            hist = hist_real + 1j * hist_imag
        else:
            hist = hist_real
    return hist.astype(hist_dtype, copy=False) if hist_dtype is not None else hist


@dataclass(frozen=True)
class BinnedStructure:
    """Atoms binned by element, radius, axial position, and azimuth."""

    hist: np.ndarray
    r_centers: np.ndarray
    z_centers: np.ndarray
    beta_centers: np.ndarray
    elements: tuple[str, ...]
    r_edges: np.ndarray
    z_edges: np.ndarray
    beta_edges: np.ndarray

    @property
    def n_phi(self) -> int:
        return int(self.beta_centers.size)

    @property
    def r_max(self) -> float:
        return float(self.r_edges[-1])

    @property
    def z_min(self) -> float:
        return float(self.z_edges[0])

    @property
    def z_max(self) -> float:
        return float(self.z_edges[-1])


@dataclass(frozen=True)
class SparseBinnedStructure(BinnedStructure):
    """Sparse cylindrical bins without allocating the dense 4-D histogram.

    ``hist`` is an empty dtype marker.  The active arrays are sorted in C-order
    over ``(element, R, z, beta)`` and are consumed by sparse-source solvers.
    Dense FFT methods intentionally reject this representation.
    """

    active_e: np.ndarray
    active_r: np.ndarray
    active_z: np.ndarray
    active_beta: np.ndarray
    active_values: np.ndarray

    @property
    def sparse_storage_nbytes(self) -> int:
        return int(
            self.active_e.nbytes
            + self.active_r.nbytes
            + self.active_z.nbytes
            + self.active_beta.nbytes
            + self.active_values.nbytes
        )


def make_sparse_cylindrical_histogram_from_flat_indices(
    flat_indices: np.ndarray,
    *,
    n_elements: int = 1,
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float,
    z_range: tuple[float, float],
    element_order: Iterable[str] | None = None,
    atom_weights: np.ndarray | None = None,
    value_dtype: np.dtype | str = np.float32,
) -> SparseBinnedStructure:
    """Aggregate flat atom-bin indices into a sparse cylindrical structure."""

    flat = np.asarray(flat_indices)
    if flat.ndim != 1 or not np.issubdtype(flat.dtype, np.integer):
        raise ValueError("flat_indices must be a one-dimensional integer array")
    n_elements = int(n_elements)
    if n_elements <= 0:
        raise ValueError("n_elements must be positive")
    n_bins = n_elements * int(n_r) * int(n_z) * int(n_phi)
    flat64 = np.ascontiguousarray(flat, dtype=np.int64)
    if flat64.size and (
        flat64.min(initial=0) < 0 or flat64.max(initial=0) >= n_bins
    ):
        raise ValueError("flat_indices contain an out-of-range bin")

    out_dtype = _normalize_hist_dtype(value_dtype)
    if out_dtype is None or out_dtype.kind not in {"f", "c"}:
        raise ValueError("value_dtype must be float32, float64, complex64, or complex128")
    if atom_weights is None:
        unique, counts = np.unique(flat64, return_counts=True)
        values = counts.astype(out_dtype, copy=False)
    else:
        weights = np.asarray(atom_weights)
        if weights.shape != flat64.shape:
            raise ValueError("atom_weights must have one entry per flat index")
        if np.iscomplexobj(weights) and out_dtype.kind != "c":
            raise ValueError("complex atom_weights require a complex value_dtype")
        order = np.argsort(flat64, kind="stable")
        sorted_flat = flat64[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_flat)) + 1]
        unique = sorted_flat[starts]
        values = np.add.reduceat(weights[order], starts).astype(out_dtype, copy=False)

    work = unique.copy()
    active_beta = np.ascontiguousarray(work % n_phi, dtype=np.intp)
    work //= n_phi
    active_z = np.ascontiguousarray(work % n_z, dtype=np.intp)
    work //= n_z
    active_r = np.ascontiguousarray(work % n_r, dtype=np.intp)
    work //= n_r
    active_e = np.ascontiguousarray(work, dtype=np.intp)

    if element_order is None:
        ordered_elements = tuple(str(i) for i in range(n_elements))
    else:
        ordered_elements = tuple(str(e) for e in element_order)
        if len(ordered_elements) != n_elements:
            raise ValueError("element_order must have n_elements entries")
    (
        r_edges,
        z_edges,
        beta_edges,
        r_centers,
        z_centers,
        beta_centers,
    ) = _regular_axes(
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
    )
    return SparseBinnedStructure(
        hist=np.empty(0, dtype=out_dtype),
        r_centers=r_centers,
        z_centers=z_centers,
        beta_centers=beta_centers,
        elements=ordered_elements,
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
        active_e=active_e,
        active_r=active_r,
        active_z=active_z,
        active_beta=active_beta,
        active_values=np.ascontiguousarray(values, dtype=out_dtype),
    )


def make_cylindrical_histogram(
    coords: np.ndarray,
    elements: Iterable[str] | None = None,
    *,
    element_indices: np.ndarray | None = None,
    n_elements: int | None = None,
    atom_weights: np.ndarray | None = None,
    binning_dtype: np.dtype | str | None = None,
    hist_dtype: np.dtype | str | None = None,
    backend: str = "numpy",
    angle_lut_size: int = 0,
    angle_lut_mode: str = "nearest",
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float | None = None,
    z_range: tuple[float, float] | None = None,
    element_order: Iterable[str] | None = None,
) -> BinnedStructure:
    """Bin atoms onto a cylindrical ``(element, R, z, beta)`` histogram."""

    hist_dtype = _normalize_hist_dtype(hist_dtype)
    coords = np.asarray(coords)
    if binning_dtype is not None:
        coords = coords.astype(binning_dtype, copy=False)
    elif not np.issubdtype(coords.dtype, np.floating):
        coords = coords.astype(float, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if n_r <= 0 or n_z <= 0 or n_phi <= 0:
        raise ValueError("n_r, n_z, and n_phi must be positive")
    if backend not in {"numpy", "numba", "numba-parallel", "cpp"}:
        raise ValueError(
            "backend must be 'numpy', 'numba', 'numba-parallel', or 'cpp'"
        )
    angle_lut_size = int(angle_lut_size)
    if angle_lut_size < 0:
        raise ValueError("angle_lut_size must be non-negative")
    if angle_lut_size and backend != "cpp":
        raise ValueError("angle_lut_size is only supported by backend='cpp'")
    if angle_lut_mode not in {"nearest", "cubic"}:
        raise ValueError("angle_lut_mode must be 'nearest' or 'cubic'")

    n_atoms = coords.shape[0]
    if elements is not None and element_indices is not None:
        raise ValueError("provide either elements or element_indices, not both")

    single_default_element = elements is None and element_indices is None
    if element_indices is not None:
        element_indices = np.asarray(element_indices)
        if element_indices.shape != (n_atoms,):
            raise ValueError("element_indices must have one entry per atom")
        if not np.issubdtype(element_indices.dtype, np.integer):
            raise ValueError("element_indices must be an integer array")
        if n_elements is None:
            n_element_value = int(element_indices.max(initial=-1)) + 1
            n_element_value = max(n_element_value, 1)
        else:
            n_element_value = int(n_elements)
        if n_element_value <= 0:
            raise ValueError("n_elements must be positive")
        if element_indices.size and (
            element_indices.min(initial=0) < 0
            or element_indices.max(initial=0) >= n_element_value
        ):
            raise ValueError("element_indices must be in [0, n_elements)")
        if element_order is None:
            ordered_elements = tuple(str(i) for i in range(n_element_value))
        else:
            ordered_elements = tuple(str(e) for e in element_order)
            if len(ordered_elements) != n_element_value:
                raise ValueError("element_order must have n_elements entries")
    elif single_default_element:
        if n_elements not in (None, 1):
            raise ValueError("n_elements requires element_indices")
        ordered_elements = ("X",)
        element_indices = None
    else:
        if n_elements is not None:
            raise ValueError("n_elements requires element_indices")
        element_indices, ordered_elements = encode_elements(
            elements,
            element_order=element_order,
            dtype=np.intp,
        )
        if element_indices.shape != (n_atoms,):
            raise ValueError("elements must have one entry per atom")

    weights = None
    if atom_weights is not None:
        weights = np.asarray(atom_weights)
        if weights.shape != (n_atoms,):
            raise ValueError("atom_weights must have one entry per atom")
    if weights is None:
        if hist_dtype is not None and hist_dtype.kind == "c":
            raise ValueError("complex hist_dtype requires atom_weights")
    else:
        has_imaginary_weights = np.iscomplexobj(weights) and bool(np.any(weights.imag))
        if has_imaginary_weights:
            if hist_dtype is not None and hist_dtype.kind != "c":
                raise ValueError("complex atom_weights require complex hist_dtype")
        elif hist_dtype is not None and hist_dtype.kind not in {"f", "c"}:
            raise ValueError("weighted histograms require float or complex hist_dtype")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    need_numpy_radius = r_max is None or backend not in {"numba", "cpp"}
    radius = np.sqrt(x * x + y * y) if need_numpy_radius else None

    if r_max is None:
        if radius is None:
            radius = np.sqrt(x * x + y * y)
        r_max = float(radius.max(initial=0.0))
        r_max = max(r_max, 1e-12) * (1.0 + 1e-12)
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    if z_range is None:
        z_min = float(z.min(initial=0.0))
        z_max = float(z.max(initial=0.0))
        if z_min == z_max:
            z_min -= 0.5
            z_max += 0.5
        else:
            pad = 1e-12 * max(1.0, abs(z_min), abs(z_max))
            z_min -= pad
            z_max += pad
    else:
        z_min, z_max = map(float, z_range)
    if z_min >= z_max:
        raise ValueError("z_range must be increasing")

    r_edges = np.linspace(0.0, r_max, n_r + 1)
    z_edges = np.linspace(z_min, z_max, n_z + 1)
    beta_edges = np.linspace(0.0, _TWO_PI, n_phi + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])

    if backend == "cpp":
        try:
            from . import _cpp_histogram
        except ImportError as exc:
            raise ImportError(
                "backend='cpp' requires the pybind11 extension to be built. "
                "Install build dependencies and run `python setup.py build_ext --inplace` "
                "or `python -m pip install -e .`."
            ) from exc

        coords_cpp = np.ascontiguousarray(coords)
        element_indices_cpp = (
            None
            if element_indices is None
            else np.ascontiguousarray(element_indices, dtype=np.int64)
        )
        if weights is None:
            if hist_dtype == np.dtype("uint32"):
                hist = _cpp_histogram.histogram_unweighted_uint32(
                    coords_cpp,
                    element_indices_cpp,
                    len(ordered_elements),
                    n_r,
                    n_z,
                    n_phi,
                    r_max,
                    z_min,
                    z_max,
                    angle_lut_size,
                    angle_lut_mode,
                )
            elif hist_dtype == np.dtype("float32"):
                hist = _cpp_histogram.histogram_unweighted_float32(
                    coords_cpp,
                    element_indices_cpp,
                    len(ordered_elements),
                    n_r,
                    n_z,
                    n_phi,
                    r_max,
                    z_min,
                    z_max,
                    angle_lut_size,
                    angle_lut_mode,
                )
            else:
                hist = _cpp_histogram.histogram_unweighted(
                    coords_cpp,
                    element_indices_cpp,
                    len(ordered_elements),
                    n_r,
                    n_z,
                    n_phi,
                    r_max,
                    z_min,
                    z_max,
                    angle_lut_size,
                    angle_lut_mode,
                )
                if hist_dtype is not None and hist.dtype != hist_dtype:
                    hist = hist.astype(hist_dtype, copy=False)
        else:
            if np.iscomplexobj(weights) and np.any(weights.imag):
                if hist_dtype == np.dtype("complex64"):
                    hist = _cpp_histogram.histogram_weighted_complex64(
                        coords_cpp,
                        element_indices_cpp,
                        np.ascontiguousarray(weights, dtype=np.complex64),
                        len(ordered_elements),
                        n_r,
                        n_z,
                        n_phi,
                        r_max,
                        z_min,
                        z_max,
                        angle_lut_size,
                        angle_lut_mode,
                    )
                else:
                    weights = np.asarray(weights, dtype=np.complex128)
                    hist = _cpp_histogram.histogram_weighted_complex(
                        coords_cpp,
                        element_indices_cpp,
                        np.ascontiguousarray(weights),
                        len(ordered_elements),
                        n_r,
                        n_z,
                        n_phi,
                        r_max,
                        z_min,
                        z_max,
                        angle_lut_size,
                        angle_lut_mode,
                    )
            else:
                if hist_dtype == np.dtype("float32"):
                    hist = _cpp_histogram.histogram_weighted_real_float32(
                        coords_cpp,
                        element_indices_cpp,
                        np.ascontiguousarray(weights.real, dtype=np.float32),
                        len(ordered_elements),
                        n_r,
                        n_z,
                        n_phi,
                        r_max,
                        z_min,
                        z_max,
                        angle_lut_size,
                        angle_lut_mode,
                    )
                else:
                    hist = _cpp_histogram.histogram_weighted_real(
                        coords_cpp,
                        element_indices_cpp,
                        np.ascontiguousarray(weights.real, dtype=np.float64),
                        len(ordered_elements),
                        n_r,
                        n_z,
                        n_phi,
                        r_max,
                        z_min,
                        z_max,
                        angle_lut_size,
                        angle_lut_mode,
                    )
                    if hist_dtype is not None and hist.dtype != hist_dtype:
                        hist = hist.astype(hist_dtype, copy=False)

        return BinnedStructure(
            hist=hist,
            r_centers=r_centers,
            z_centers=z_centers,
            beta_centers=beta_centers,
            elements=ordered_elements,
            r_edges=r_edges,
            z_edges=z_edges,
            beta_edges=beta_edges,
        )

    use_numba = (
        backend in {"numba", "numba-parallel"}
        and single_default_element
        and atom_weights is None
        and element_order is None
    )
    if use_numba:
        from ._numba_histogram import histogram_single_unweighted

        hist = histogram_single_unweighted(
            coords,
            n_r=n_r,
            n_z=n_z,
            n_phi=n_phi,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
            parallel=backend == "numba-parallel",
        ).reshape(1, n_r, n_z, n_phi)
        if hist_dtype is not None and hist.dtype != hist_dtype:
            hist = hist.astype(hist_dtype, copy=False)

        return BinnedStructure(
            hist=hist,
            r_centers=r_centers,
            z_centers=z_centers,
            beta_centers=beta_centers,
            elements=ordered_elements,
            r_edges=r_edges,
            z_edges=z_edges,
            beta_edges=beta_edges,
        )

    if radius is None:
        radius = np.sqrt(x * x + y * y)
    radius *= n_r / r_max
    r_idx = radius.astype(np.intp, copy=False)
    z_idx = ((z - z_min) * (n_z / (z_max - z_min))).astype(np.intp, copy=False)
    np.clip(r_idx, 0, n_r - 1, out=r_idx)
    np.clip(z_idx, 0, n_z - 1, out=z_idx)

    beta = np.arctan2(y, x)
    beta[beta < 0.0] += _TWO_PI
    beta *= n_phi / _TWO_PI
    beta_idx = beta.astype(np.intp, copy=False)
    np.clip(beta_idx, 0, n_phi - 1, out=beta_idx)

    flat_idx = r_idx
    flat_idx *= n_z
    flat_idx += z_idx
    flat_idx *= n_phi
    flat_idx += beta_idx
    if element_indices is not None:
        flat_idx += element_indices * (n_r * n_z * n_phi)
    n_bins = len(ordered_elements) * n_r * n_z * n_phi
    if weights is None:
        hist = np.bincount(flat_idx, minlength=n_bins)
    else:
        weights = np.asarray(weights, dtype=np.complex128)
        hist_real = np.bincount(flat_idx, weights=weights.real, minlength=n_bins)
        if np.any(weights.imag):
            hist_imag = np.bincount(flat_idx, weights=weights.imag, minlength=n_bins)
            hist = hist_real + 1j * hist_imag
        else:
            hist = hist_real
    hist = hist.reshape(len(ordered_elements), n_r, n_z, n_phi)
    if hist_dtype is not None and hist.dtype != hist_dtype:
        hist = hist.astype(hist_dtype, copy=False)

    return BinnedStructure(
        hist=hist,
        r_centers=r_centers,
        z_centers=z_centers,
        beta_centers=beta_centers,
        elements=ordered_elements,
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )


def make_cylindrical_histogram_indexed(
    coords: np.ndarray,
    element_indices: np.ndarray | None = None,
    *,
    n_elements: int | None = None,
    element_order: Iterable[str] | None = None,
    atom_weights: np.ndarray | None = None,
    binning_dtype: np.dtype | str | None = None,
    hist_dtype: np.dtype | str | None = None,
    backend: str = "cpp",
    angle_lut_size: int = 0,
    angle_lut_mode: str = "nearest",
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float | None = None,
    z_range: tuple[float, float] | None = None,
) -> BinnedStructure:
    """Bin atoms when element labels are already encoded as integer indices."""

    return make_cylindrical_histogram(
        coords,
        element_indices=element_indices,
        n_elements=n_elements,
        atom_weights=atom_weights,
        binning_dtype=binning_dtype,
        hist_dtype=hist_dtype,
        backend=backend,
        angle_lut_size=angle_lut_size,
        angle_lut_mode=angle_lut_mode,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        element_order=element_order,
    )


def cylindrical_flat_indices(
    coords: np.ndarray,
    *,
    element_indices: np.ndarray | None = None,
    n_elements: int | None = None,
    binning_dtype: np.dtype | str | None = None,
    backend: str = "cpp",
    angle_lut_size: int = 0,
    angle_lut_mode: str = "nearest",
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float | None = None,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return flat C-order ``(element, R, z, beta)`` bin indices for atoms."""

    if backend not in {"numpy", "cpp"}:
        raise ValueError("backend must be 'numpy' or 'cpp'")
    if angle_lut_size < 0:
        raise ValueError("angle_lut_size must be non-negative")
    if angle_lut_size and backend != "cpp":
        raise ValueError("angle_lut_size is only supported by backend='cpp'")
    if angle_lut_mode not in {"nearest", "cubic"}:
        raise ValueError("angle_lut_mode must be 'nearest' or 'cubic'")

    coords = np.asarray(coords)
    if binning_dtype is not None:
        coords = coords.astype(binning_dtype, copy=False)
    elif not np.issubdtype(coords.dtype, np.floating):
        coords = coords.astype(float, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if n_r <= 0 or n_z <= 0 or n_phi <= 0:
        raise ValueError("n_r, n_z, and n_phi must be positive")

    n_atoms = coords.shape[0]
    if element_indices is None:
        n_element_value = 1 if n_elements is None else int(n_elements)
        if n_element_value != 1:
            raise ValueError("n_elements > 1 requires element_indices")
        element_indices_checked = None
    else:
        element_indices_arr = np.asarray(element_indices)
        if element_indices_arr.shape != (n_atoms,):
            raise ValueError("element_indices must have one entry per atom")
        if not np.issubdtype(element_indices_arr.dtype, np.integer):
            raise ValueError("element_indices must be an integer array")
        if n_elements is None:
            n_element_value = int(element_indices_arr.max(initial=-1)) + 1
            n_element_value = max(n_element_value, 1)
        else:
            n_element_value = int(n_elements)
        if n_element_value <= 0:
            raise ValueError("n_elements must be positive")
        if element_indices_arr.size and (
            element_indices_arr.min(initial=0) < 0
            or element_indices_arr.max(initial=0) >= n_element_value
        ):
            raise ValueError("element_indices must be in [0, n_elements)")
        element_indices_checked = np.ascontiguousarray(element_indices_arr, dtype=np.int64)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    radius = np.sqrt(x * x + y * y) if r_max is None or backend == "numpy" else None
    if r_max is None:
        if radius is None:
            radius = np.sqrt(x * x + y * y)
        r_max = float(radius.max(initial=0.0))
        r_max = max(r_max, 1e-12) * (1.0 + 1e-12)
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    if z_range is None:
        z_min = float(z.min(initial=0.0))
        z_max = float(z.max(initial=0.0))
        if z_min == z_max:
            z_min -= 0.5
            z_max += 0.5
        else:
            pad = 1e-12 * max(1.0, abs(z_min), abs(z_max))
            z_min -= pad
            z_max += pad
    else:
        z_min, z_max = map(float, z_range)
    if z_min >= z_max:
        raise ValueError("z_range must be increasing")

    if backend == "cpp":
        try:
            from . import _cpp_histogram
        except ImportError as exc:
            raise ImportError(
                "backend='cpp' requires the pybind11 extension to be built. "
                "Install build dependencies and run `python setup.py build_ext --inplace` "
                "or `python -m pip install -e .`."
            ) from exc
        return _cpp_histogram.flat_indices_from_coords(
            np.ascontiguousarray(coords),
            element_indices_checked,
            n_element_value,
            n_r,
            n_z,
            n_phi,
            r_max,
            z_min,
            z_max,
            int(angle_lut_size),
            angle_lut_mode,
        )

    if radius is None:
        radius = np.sqrt(x * x + y * y)
    r_idx = (radius * (n_r / r_max)).astype(np.int64)
    z_idx = ((z - z_min) * (n_z / (z_max - z_min))).astype(np.int64)
    beta = np.arctan2(y, x)
    beta[beta < 0.0] += _TWO_PI
    beta_idx = (beta * (n_phi / _TWO_PI)).astype(np.int64)
    np.clip(r_idx, 0, n_r - 1, out=r_idx)
    np.clip(z_idx, 0, n_z - 1, out=z_idx)
    np.clip(beta_idx, 0, n_phi - 1, out=beta_idx)

    if element_indices_checked is None:
        flat_idx = r_idx
    else:
        flat_idx = np.array(element_indices_checked, dtype=np.int64, copy=True)
        flat_idx *= n_r
        flat_idx += r_idx
    flat_idx *= n_z
    flat_idx += z_idx
    flat_idx *= n_phi
    flat_idx += beta_idx
    return flat_idx


def make_cylindrical_histogram_from_flat_indices(
    flat_indices: np.ndarray,
    *,
    n_elements: int = 1,
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float,
    z_range: tuple[float, float],
    element_order: Iterable[str] | None = None,
    atom_weights: np.ndarray | None = None,
    hist_dtype: np.dtype | str | None = None,
    backend: str = "cpp",
    validate_indices: bool = True,
) -> BinnedStructure:
    """Build a cylindrical histogram from precomputed flat bin indices.

    ``flat_indices`` are interpreted in C-order over
    ``(element, R, z, beta)``. This path skips all coordinate transforms and is
    useful when bin indices are reused across repeated weighted histograms.
    Set ``validate_indices=False`` only for trusted indices produced by this
    package or by an already validated caller.
    """

    hist_dtype = _normalize_hist_dtype(hist_dtype)
    n_elements = int(n_elements)
    if n_elements <= 0:
        raise ValueError("n_elements must be positive")
    if element_order is None:
        ordered_elements = ("X",) if n_elements == 1 else tuple(
            str(i) for i in range(n_elements)
        )
    else:
        ordered_elements = tuple(str(e) for e in element_order)
        if len(ordered_elements) != n_elements:
            raise ValueError("element_order must have n_elements entries")

    (
        r_edges,
        z_edges,
        beta_edges,
        r_centers,
        z_centers,
        beta_centers,
    ) = _regular_axes(
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
    )
    n_bins = n_elements * n_r * n_z * n_phi
    hist = _histogram_flat_indices(
        flat_indices,
        n_bins=n_bins,
        atom_weights=atom_weights,
        hist_dtype=hist_dtype,
        backend=backend,
        validate_indices=validate_indices,
    ).reshape(n_elements, n_r, n_z, n_phi)

    return BinnedStructure(
        hist=hist,
        r_centers=r_centers,
        z_centers=z_centers,
        beta_centers=beta_centers,
        elements=ordered_elements,
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )


def make_cylindrical_histogram_from_indices(
    r_indices: np.ndarray,
    z_indices: np.ndarray,
    beta_indices: np.ndarray,
    *,
    element_indices: np.ndarray | None = None,
    n_elements: int | None = None,
    n_r: int = 32,
    n_z: int = 32,
    n_phi: int = 180,
    r_max: float,
    z_range: tuple[float, float],
    element_order: Iterable[str] | None = None,
    atom_weights: np.ndarray | None = None,
    hist_dtype: np.dtype | str | None = None,
    backend: str = "cpp",
) -> BinnedStructure:
    """Build a cylindrical histogram from precomputed ``R/z/beta`` indices."""

    r_idx = np.asarray(r_indices)
    z_idx = np.asarray(z_indices)
    beta_idx = np.asarray(beta_indices)
    if r_idx.shape != z_idx.shape or r_idx.shape != beta_idx.shape:
        raise ValueError("r_indices, z_indices, and beta_indices must have the same shape")
    if r_idx.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if not (
        np.issubdtype(r_idx.dtype, np.integer)
        and np.issubdtype(z_idx.dtype, np.integer)
        and np.issubdtype(beta_idx.dtype, np.integer)
    ):
        raise ValueError("indices must be integer arrays")
    if r_idx.size and (
        r_idx.min(initial=0) < 0
        or r_idx.max(initial=0) >= n_r
        or z_idx.min(initial=0) < 0
        or z_idx.max(initial=0) >= n_z
        or beta_idx.min(initial=0) < 0
        or beta_idx.max(initial=0) >= n_phi
    ):
        raise ValueError("indices contain an out-of-range bin")

    if element_indices is None:
        n_element_value = 1 if n_elements is None else int(n_elements)
        if n_element_value != 1:
            raise ValueError("n_elements > 1 requires element_indices")
        flat_idx = np.array(r_idx, dtype=np.int64, copy=True)
    else:
        elem_idx = np.asarray(element_indices)
        if elem_idx.shape != r_idx.shape:
            raise ValueError("element_indices must have one entry per atom")
        if not np.issubdtype(elem_idx.dtype, np.integer):
            raise ValueError("element_indices must be an integer array")
        if n_elements is None:
            n_element_value = int(elem_idx.max(initial=-1)) + 1
            n_element_value = max(n_element_value, 1)
        else:
            n_element_value = int(n_elements)
        if n_element_value <= 0:
            raise ValueError("n_elements must be positive")
        if elem_idx.size and (
            elem_idx.min(initial=0) < 0 or elem_idx.max(initial=0) >= n_element_value
        ):
            raise ValueError("element_indices must be in [0, n_elements)")
        flat_idx = np.array(elem_idx, dtype=np.int64, copy=True)
        flat_idx *= n_r
        flat_idx += r_idx

    flat_idx *= n_z
    flat_idx += z_idx
    flat_idx *= n_phi
    flat_idx += beta_idx

    return make_cylindrical_histogram_from_flat_indices(
        flat_idx,
        n_elements=n_element_value,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        element_order=element_order,
        atom_weights=atom_weights,
        hist_dtype=hist_dtype,
        backend=backend,
        validate_indices=False,
    )
