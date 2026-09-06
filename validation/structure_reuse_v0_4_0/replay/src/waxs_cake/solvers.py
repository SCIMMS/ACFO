"""Direct, circular-FFT, Jacobi-Anger, hybrid, and NUFFT WAXS solvers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import fft as scipy_fft
from scipy import special

from .geometry import ewald_ring
from .histogram import BinnedStructure, SparseBinnedStructure

FormFactors = (
    Mapping[str, complex | float | np.ndarray | Callable[[np.ndarray], np.ndarray]]
    | Callable[[str, np.ndarray], np.ndarray]
    | None
)
CircularBackend = str


def uniform_phi_grid(n_phi: int) -> np.ndarray:
    return (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)


def normalize_form_factors(
    elements: Sequence[str],
    q: np.ndarray,
    form_factors: FormFactors = None,
) -> np.ndarray:
    """Return an ``(n_elements, n_q)`` complex form-factor array."""

    q = np.asarray(q, dtype=float)
    out = np.ones((len(elements), q.size), dtype=np.complex128)
    if form_factors is None:
        return out

    if callable(form_factors):
        for i, element in enumerate(elements):
            out[i] = np.asarray(form_factors(element, q), dtype=np.complex128)
        return out

    for i, element in enumerate(elements):
        value = form_factors.get(element, 1.0)
        if callable(value):
            out[i] = np.asarray(value(q), dtype=np.complex128)
        else:
            arr = np.asarray(value, dtype=np.complex128)
            out[i] = arr if arr.ndim else np.full(q.size, arr, dtype=np.complex128)
    return out


def direct_amplitude(
    coords: np.ndarray,
    q: np.ndarray,
    wavelength: float,
    phi: np.ndarray,
    *,
    elements: Sequence[str] | None = None,
    form_factors: FormFactors = None,
    atom_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reference direct phase-sum amplitude on a WAXS cake grid."""

    coords = np.asarray(coords, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")

    n_atoms = coords.shape[0]
    if elements is None:
        atom_elements = np.full(n_atoms, "X", dtype=object)
    else:
        atom_elements = np.asarray(list(elements), dtype=object)
        if atom_elements.shape != (n_atoms,):
            raise ValueError("elements must have one entry per atom")

    if atom_weights is None:
        weights = np.ones(n_atoms, dtype=np.complex128)
    else:
        weights = np.asarray(atom_weights, dtype=np.complex128)
        if weights.shape != (n_atoms,):
            raise ValueError("atom_weights must have one entry per atom")

    ordered_elements = tuple(dict.fromkeys(str(e) for e in atom_elements))
    element_to_index = {element: i for i, element in enumerate(ordered_elements)}
    atom_element_indices = np.array([element_to_index[str(e)] for e in atom_elements])
    ff = normalize_form_factors(ordered_elements, q, form_factors)

    q_perp, q_z = ewald_ring(q, wavelength)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    out = np.empty((q.size, phi.size), dtype=np.complex128)
    xy_projection = x[:, None] * cos_phi[None, :] + y[:, None] * sin_phi[None, :]
    for iq in range(q.size):
        coeff = weights * ff[atom_element_indices, iq]
        phase = q_perp[iq] * xy_projection + q_z[iq] * z[:, None]
        out[iq] = np.sum(coeff[:, None] * np.exp(1j * phase), axis=0)
    return out


def circular_fft_amplitude(
    binned: BinnedStructure,
    q: np.ndarray,
    wavelength: float,
    *,
    phi: np.ndarray | None = None,
    form_factors: FormFactors = None,
    q_block_size: int = 128,
    cache_kernel_fft: bool = False,
    kernel_interpolation_dx: float | None = None,
    circular_backend: CircularBackend = "auto",
    harmonic_bandlimit_margin: int | None = None,
    complex_dtype: np.dtype | str | None = None,
) -> np.ndarray:
    """Binned Ewald-ring circular convolution using sampled kernels and FFTs."""

    plan = PreparedCakePlan(
        binned,
        q,
        wavelength,
        phi=phi,
        form_factors=form_factors,
        q_block_size=q_block_size,
        cache_kernel_fft=cache_kernel_fft,
        kernel_interpolation_dx=kernel_interpolation_dx,
        circular_backend=circular_backend,
        harmonic_bandlimit_margin=harmonic_bandlimit_margin,
        complex_dtype=complex_dtype,
    )
    return plan.circular_fft()


def estimate_bessel_cutoff(
    x: float,
    *,
    tol: float = 1e-8,
    n_phi: int | None = None,
    consecutive: int = 8,
    safety_margin: int = 4,
) -> int:
    """Estimate the harmonic cutoff needed for ``exp(i*x*cos(theta))``.

    The cutoff is capped at ``n_phi // 2`` when an angular grid size is supplied.
    This cap is a sampling limit, not a guarantee that unresolved high harmonics
    are physically negligible.
    """

    x = float(abs(x))
    if x == 0.0:
        return 0
    cap = None if n_phi is None else int(n_phi // 2)
    max_order = max(int(np.ceil(x + 10.0 * np.cbrt(x + 1.0) + 32.0)), consecutive + 1)
    if cap is not None:
        max_order = max(max_order, cap)

    quiet = 0
    chosen = max_order
    for order in range(max(0, int(np.floor(x)) - safety_margin), max_order + 1):
        if abs(special.jv(order, x)) < tol:
            quiet += 1
            if quiet >= consecutive:
                chosen = max(0, order - consecutive + 1)
                break
        else:
            quiet = 0

    if cap is not None:
        chosen = min(chosen + safety_margin, cap)
    else:
        chosen += safety_margin
    return int(chosen)


def _harmonic_coefficients(binned: BinnedStructure, modes: np.ndarray) -> np.ndarray:
    delta = 2.0 * np.pi / binned.n_phi
    hhat = np.fft.fft(binned.hist, axis=-1)
    indices = np.mod(modes, binned.n_phi)
    coeff = np.take(hhat, indices, axis=-1)
    return coeff * np.exp(-0.5j * delta * modes)


def _normalize_q_indices(q_indices: np.ndarray | None, n_q: int) -> np.ndarray:
    if q_indices is None:
        return np.arange(n_q)
    indices = np.asarray(q_indices)
    if indices.dtype == bool:
        if indices.shape != (n_q,):
            raise ValueError("boolean q_indices must have one entry per q")
        return np.flatnonzero(indices)
    return indices.astype(int, copy=False)


def _unit_complex_from_phase(phase: np.ndarray) -> np.ndarray:
    out = np.empty(phase.shape, dtype=np.complex128)
    out.real = np.cos(phase)
    out.imag = np.sin(phase)
    return out


def _unit_complex_from_phase_dtype(
    phase: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("complex64"):
        phase = np.asarray(phase, dtype=np.float32)
        out = np.empty(phase.shape, dtype=np.complex64)
    elif dtype == np.dtype("complex128"):
        phase = np.asarray(phase, dtype=np.float64)
        out = np.empty(phase.shape, dtype=np.complex128)
    else:
        raise ValueError("complex dtype must be complex64 or complex128")
    out.real = np.cos(phase)
    out.imag = np.sin(phase)
    return out


def _infer_complex_dtype(
    binned: BinnedStructure,
    dtype: np.dtype | str | None,
) -> np.dtype:
    if dtype is None:
        if binned.hist.dtype in (np.dtype("float32"), np.dtype("complex64")):
            return np.dtype("complex64")
        return np.dtype("complex128")
    normalized = np.dtype(dtype)
    if normalized not in (np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("complex_dtype must be complex64 or complex128")
    return normalized


def _cpp_solver_module(*, required: bool):
    try:
        from . import _cpp_solvers
    except ImportError as exc:
        if required:
            raise ImportError(
                "circular_backend='cpp' requires the pybind11 solver extension "
                "to be built. Run `python setup.py build_ext --inplace` or "
                "`python -m pip install -e .`."
            ) from exc
        return None
    return _cpp_solvers


class PreparedCakePlan:
    """Reusable cache for repeated WAXS cake-map evaluations on one grid."""

    def __init__(
        self,
        binned: BinnedStructure,
        q: np.ndarray,
        wavelength: float,
        *,
        phi: np.ndarray | None = None,
        form_factors: FormFactors = None,
        cutoff_tol: float = 1e-8,
        switch_fraction: float = 0.5,
        q_block_size: int = 128,
        cache_kernel_fft: bool = False,
        kernel_interpolation_dx: float | None = None,
        circular_backend: CircularBackend = "auto",
        harmonic_bandlimit_margin: int | None = None,
        analytic_kernel_table_dx: float | None = None,
        complex_dtype: np.dtype | str | None = None,
        q_perp: np.ndarray | None = None,
        q_z: np.ndarray | None = None,
    ) -> None:
        self.binned = binned
        self.q = np.asarray(q, dtype=float)
        self.wavelength = float(wavelength)
        self.phi = binned.beta_centers if phi is None else np.asarray(phi, dtype=float)
        self.complex_dtype = _infer_complex_dtype(binned, complex_dtype)
        self.cutoff_tol = float(cutoff_tol)
        self.switch_fraction = float(switch_fraction)
        self.q_block_size = int(q_block_size)
        if self.q_block_size <= 0:
            raise ValueError("q_block_size must be positive")
        if cache_kernel_fft and kernel_interpolation_dx is not None:
            raise ValueError(
                "cache_kernel_fft and kernel_interpolation_dx are mutually exclusive"
            )
        if kernel_interpolation_dx is not None and kernel_interpolation_dx <= 0:
            raise ValueError("kernel_interpolation_dx must be positive")
        if circular_backend not in {"auto", "numpy", "cpp"}:
            raise ValueError("circular_backend must be 'auto', 'numpy', or 'cpp'")
        if harmonic_bandlimit_margin is not None and harmonic_bandlimit_margin < 0:
            raise ValueError("harmonic_bandlimit_margin must be non-negative")
        if analytic_kernel_table_dx is not None and analytic_kernel_table_dx <= 0:
            raise ValueError("analytic_kernel_table_dx must be positive")

        if (q_perp is None) != (q_z is None):
            raise ValueError("q_perp and q_z must be provided together")
        if q_perp is None:
            self.q_perp, self.q_z = ewald_ring(self.q, self.wavelength)
        else:
            self.q_perp = np.asarray(q_perp, dtype=float)
            self.q_z = np.asarray(q_z, dtype=float)
            if self.q_perp.shape != self.q.shape or self.q_z.shape != self.q.shape:
                raise ValueError("q_perp and q_z must have the same shape as q")
        self.circular_backend = circular_backend
        self.harmonic_bandlimit_margin = harmonic_bandlimit_margin
        self.analytic_kernel_table_dx = analytic_kernel_table_dx
        self.form_factors = normalize_form_factors(
            binned.elements, self.q, form_factors
        ).astype(self.complex_dtype, copy=False)
        self.z_phase = _unit_complex_from_phase_dtype(
            self.q_z[:, None] * binned.z_centers[None, :],
            self.complex_dtype,
        )
        self._hhat: np.ndarray | None = None
        self._hhat_half: np.ndarray | None = None
        self._hhat_half_mode_cache: dict[int, np.ndarray] = {}

        self.delta_phi = 2.0 * np.pi / binned.n_phi
        self.kernel_angles = np.arange(binned.n_phi) * self.delta_phi
        self.cos_kernel_angles = np.cos(self.kernel_angles)
        self.n_half = binned.n_phi // 2

        self._kernel_hat: np.ndarray | None = None
        self._kernel_table: KernelInterpolationTable | None = None
        self._bessel_kernel_tables: dict[float, tuple[int, np.ndarray]] = {}
        self._z_reduced: np.ndarray | None = None
        self._q_radius: np.ndarray | None = None
        self._cutoffs: np.ndarray | None = None
        self._harmonic_max = -1
        self._modes = np.empty(0, dtype=int)
        self._hcoef = np.empty(0, dtype=np.complex128)
        self._phi_basis = np.empty(0, dtype=np.complex128)
        self._i_to_m = np.empty(0, dtype=np.complex128)
        self._sparse_rz_cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None
        self._sparse_flat_cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None
        self._sparse_profile_cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None
        self._sparse_er_profile_cache: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None
        self._sparse_twiddle_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._sparse_profile_hhat_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._adaptive_profile_hhat_cache: dict[
            tuple[tuple[int, ...], float],
            np.ndarray,
        ] = {}
        self._adaptive_profile_hhat_stats: dict[
            tuple[tuple[int, ...], float],
            dict[str, float | int],
        ] = {}
        self._last_adaptive_profile_stats: dict[str, float | int] | None = None
        self._sparse_only = isinstance(binned, SparseBinnedStructure)
        if self._sparse_only:
            self._sparse_flat_cache = (
                np.ascontiguousarray(binned.active_e, dtype=np.intp),
                np.ascontiguousarray(binned.active_r, dtype=np.intp),
                np.ascontiguousarray(binned.active_z, dtype=np.intp),
                np.ascontiguousarray(binned.active_beta, dtype=np.intp),
                np.ascontiguousarray(binned.active_values, dtype=self.complex_dtype),
            )

        if cache_kernel_fft:
            self.precompute_circular_kernels()
        if kernel_interpolation_dx is not None:
            self._kernel_table = KernelInterpolationTable(
                self.binned.n_phi,
                float(np.max(self.q_perp) * self.binned.r_max),
                kernel_interpolation_dx,
                complex_dtype=self.complex_dtype,
            )

    @property
    def modes(self) -> np.ndarray:
        return self._modes

    @property
    def hcoef(self) -> np.ndarray:
        return self._hcoef

    @property
    def phi_basis(self) -> np.ndarray:
        return self._phi_basis

    @property
    def i_to_m(self) -> np.ndarray:
        return self._i_to_m

    @property
    def hhat(self) -> np.ndarray:
        if self._sparse_only:
            raise RuntimeError(
                "dense FFT methods are unavailable for SparseBinnedStructure; "
                "use a sparse-source solver"
            )
        if self._hhat is None:
            self._hhat = scipy_fft.fft(
                self.binned.hist,
                axis=-1,
                workers=-1,
            ).astype(self.complex_dtype, copy=False)
        return self._hhat

    @property
    def has_real_histogram(self) -> bool:
        if self._sparse_only:
            return not np.iscomplexobj(self.binned.active_values)
        return not np.iscomplexobj(self.binned.hist)

    @property
    def hhat_half(self) -> np.ndarray:
        if not self.has_real_histogram:
            raise ValueError("hhat_half is only available for real histograms")
        if self._hhat_half is None:
            if self._hhat is not None:
                self._hhat_half = np.ascontiguousarray(
                    self._hhat[..., : self.n_half + 1],
                    dtype=self.complex_dtype,
                )
            else:
                self._hhat_half = scipy_fft.rfft(
                    self.binned.hist,
                    axis=-1,
                    workers=-1,
                ).astype(self.complex_dtype, copy=False)
        return self._hhat_half

    def hhat_half_modes(self, max_h: int) -> np.ndarray:
        """Return real-histogram FFT modes ``0..max_h`` with compact row stride."""

        max_h = int(max_h)
        if max_h < 0:
            raise ValueError("max_h must be non-negative")
        if max_h > self.n_half:
            raise ValueError("max_h cannot exceed n_phi // 2")
        if max_h == self.n_half:
            return self.hhat_half

        n_modes = max_h + 1
        cached = self._hhat_half_mode_cache.get(n_modes)
        if cached is not None:
            return cached

        if self._hhat_half is None and self._hhat is not None:
            out = np.ascontiguousarray(
                self._hhat[..., :n_modes],
                dtype=self.complex_dtype,
            )
        else:
            half = self.hhat_half
            out = np.ascontiguousarray(
                half[..., :n_modes],
                dtype=self.complex_dtype,
            )
        self._hhat_half_mode_cache[n_modes] = out
        return out

    def _hhat_positive_modes_for_half_contraction(self) -> np.ndarray:
        """Return positive FFT modes for real-histogram half-spectrum contraction.

        If a full complex FFT is already cached, reuse it directly and let the
        C++ half-spectrum worker read only the positive modes it needs. This
        avoids materializing an additional compact positive-mode copy after a
        dense circular solve.
        """

        if not self.has_real_histogram:
            raise ValueError("positive half-spectrum is only available for real histograms")
        if self._hhat is not None:
            return self._hhat
        return self.hhat_half

    @property
    def q_radius(self) -> np.ndarray:
        if self._q_radius is None:
            self._q_radius = self.q_perp * self.binned.r_max
        return self._q_radius

    @property
    def cutoffs(self) -> np.ndarray:
        if self._cutoffs is None:
            self._cutoffs = np.array(
                [
                    estimate_bessel_cutoff(
                        x,
                        tol=self.cutoff_tol,
                        n_phi=self.binned.n_phi,
                    )
                    for x in self.q_radius
                ],
                dtype=int,
            )
        return self._cutoffs

    @property
    def use_jacobi(self) -> np.ndarray:
        return self.cutoffs <= int(np.floor(self.switch_fraction * self.n_half))

    def _check_circular_grid(self) -> None:
        if self.phi.shape != self.binned.beta_centers.shape or not np.allclose(
            self.phi, self.binned.beta_centers
        ):
            raise ValueError("circular FFT requires the histogram beta grid as phi")

    def _kernel_hat_block(self, indices: np.ndarray) -> np.ndarray:
        if self._kernel_hat is not None:
            return self._kernel_hat[indices]
        if self._kernel_table is not None:
            return self._kernel_table.khat(self.q_perp[indices], self.binned.r_centers)
        phase = (
            self.q_perp[indices, None, None]
            * self.binned.r_centers[None, :, None]
            * self.cos_kernel_angles[None, None, :]
        )
        kernel = _unit_complex_from_phase_dtype(phase, self.complex_dtype)
        return scipy_fft.fft(kernel, axis=-1, workers=1)

    def _kernel_hat_block_r(
        self,
        indices: np.ndarray,
        r_indices: np.ndarray,
    ) -> np.ndarray:
        if self._kernel_hat is not None:
            return np.take(self._kernel_hat[indices], r_indices, axis=1)
        r_centers = self.binned.r_centers[r_indices]
        if self._kernel_table is not None:
            return self._kernel_table.khat(self.q_perp[indices], r_centers)
        phase = (
            self.q_perp[indices, None, None]
            * r_centers[None, :, None]
            * self.cos_kernel_angles[None, None, :]
        )
        kernel = _unit_complex_from_phase_dtype(phase, self.complex_dtype)
        return scipy_fft.fft(kernel, axis=-1, workers=1)

    def _bessel_kernel_table(self, max_cutoff: int, dx: float) -> np.ndarray:
        """Return cached ``J_h(x)`` samples for ``h=0..max_cutoff+1``."""

        max_cutoff = int(max_cutoff)
        dx = float(dx)
        required_orders = max_cutoff + 2
        cached = self._bessel_kernel_tables.get(dx)
        if cached is not None:
            cached_orders, table = cached
            if cached_orders >= required_orders:
                return table[:required_orders]

        x_max = float(np.max(np.abs(self.q_perp)) * self.binned.r_max)
        n_x = int(np.ceil(x_max / dx)) + 2
        x_grid = np.arange(n_x, dtype=np.float64) * dx
        orders = np.arange(required_orders, dtype=np.int64)
        table = special.jv(orders[:, None], x_grid[None, :])
        table = np.ascontiguousarray(table, dtype=np.float64)
        self._bessel_kernel_tables[dx] = (required_orders, table)
        return table

    def _analytic_kernel_hat_modes_from_table_python(
        self,
        indices: np.ndarray,
        max_cutoff: int,
        dx: float,
        r_centers: np.ndarray | None = None,
    ) -> np.ndarray:
        table = self._bessel_kernel_table(max_cutoff, dx)
        if r_centers is None:
            r_centers = self.binned.r_centers
        x = self.q_perp[indices, None] * r_centers[None, :]
        scaled = np.clip(x / dx, 0.0, table.shape[1] - 1.0)
        lower = np.floor(scaled).astype(np.int64)
        lower = np.clip(lower, 0, table.shape[1] - 2)
        t = scaled - lower

        values0 = table[: max_cutoff + 1, lower]
        values1 = table[: max_cutoff + 1, lower + 1]
        deriv0 = np.empty_like(values0)
        deriv1 = np.empty_like(values1)
        deriv0[0] = -table[1, lower]
        deriv1[0] = -table[1, lower + 1]
        if max_cutoff > 0:
            deriv0[1:] = 0.5 * (
                table[:max_cutoff, lower] - table[2 : max_cutoff + 2, lower]
            )
            deriv1[1:] = 0.5 * (
                table[:max_cutoff, lower + 1]
                - table[2 : max_cutoff + 2, lower + 1]
            )

        t = t[None, :, :]
        t2 = t * t
        t3 = t2 * t
        interp = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * values0
            + (t3 - 2.0 * t2 + t) * dx * deriv0
            + (-2.0 * t3 + 3.0 * t2) * values1
            + (t3 - t2) * dx * deriv1
        )
        modes = np.arange(max_cutoff + 1, dtype=int)
        powers = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[
            np.mod(modes, 4)
        ]
        positive = self.binned.n_phi * interp * powers[:, None, None]
        positive = np.moveaxis(positive, 0, -1)
        return np.ascontiguousarray(positive, dtype=self.complex_dtype)

    def _analytic_kernel_hat_modes_r(
        self,
        indices: np.ndarray,
        r_indices: np.ndarray,
        max_cutoff: int,
        *,
        table_dx: float | None = None,
    ) -> np.ndarray:
        """Return positive analytic kernel coefficients for selected ``R`` bins."""

        max_cutoff = int(max_cutoff)
        if max_cutoff < 0 or max_cutoff >= self.n_half:
            raise ValueError("max_cutoff must satisfy 0 <= max_cutoff < n_phi / 2")
        r_centers = np.ascontiguousarray(
            self.binned.r_centers[r_indices],
            dtype=np.float64,
        )
        cpp_solvers = _cpp_solver_module(required=False)
        if table_dx is not None:
            table_dx = float(table_dx)
            if table_dx <= 0:
                raise ValueError("table_dx must be positive")
            if cpp_solvers is not None:
                func_name = (
                    "analytic_kernel_hat_modes_table64"
                    if self.complex_dtype == np.dtype("complex64")
                    else "analytic_kernel_hat_modes_table"
                )
                func = getattr(cpp_solvers, func_name, None)
                if func is not None:
                    return func(
                        np.ascontiguousarray(self.q_perp[indices], dtype=np.float64),
                        r_centers,
                        self._bessel_kernel_table(max_cutoff, table_dx),
                        int(self.binned.n_phi),
                        max_cutoff,
                        table_dx,
                    )
            return self._analytic_kernel_hat_modes_from_table_python(
                indices,
                max_cutoff,
                table_dx,
                r_centers=r_centers,
            )
        if cpp_solvers is not None:
            func_name = (
                "analytic_kernel_hat_modes_miller64"
                if self.complex_dtype == np.dtype("complex64")
                else "analytic_kernel_hat_modes_miller"
            )
            return getattr(cpp_solvers, func_name)(
                np.ascontiguousarray(self.q_perp[indices], dtype=np.float64),
                r_centers,
                int(self.binned.n_phi),
                max_cutoff,
                64,
            )

        modes = np.arange(max_cutoff + 1, dtype=int)
        x = self.q_perp[indices, None] * r_centers[None, :]
        powers = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[
            np.mod(modes, 4)
        ]
        positive = (
            self.binned.n_phi
            * powers[None, None, :]
            * special.jv(modes[None, None, :], x[:, :, None])
        )
        return np.ascontiguousarray(positive, dtype=self.complex_dtype)

    def _analytic_kernel_hat_modes(
        self,
        indices: np.ndarray,
        max_cutoff: int,
        *,
        table_dx: float | None = None,
    ) -> np.ndarray:
        """Return positive analytic kernel coefficients ``K_h`` for ``0 <= h <= H``."""

        max_cutoff = int(max_cutoff)
        if max_cutoff < 0 or max_cutoff >= self.n_half:
            raise ValueError("max_cutoff must satisfy 0 <= max_cutoff < n_phi / 2")
        cpp_solvers = _cpp_solver_module(required=False)
        if table_dx is not None:
            table_dx = float(table_dx)
            if table_dx <= 0:
                raise ValueError("table_dx must be positive")
            if cpp_solvers is not None:
                func_name = (
                    "analytic_kernel_hat_modes_table64"
                    if self.complex_dtype == np.dtype("complex64")
                    else "analytic_kernel_hat_modes_table"
                )
                func = getattr(cpp_solvers, func_name, None)
                if func is not None:
                    return func(
                        np.ascontiguousarray(self.q_perp[indices], dtype=np.float64),
                        np.ascontiguousarray(self.binned.r_centers, dtype=np.float64),
                        self._bessel_kernel_table(max_cutoff, table_dx),
                        int(self.binned.n_phi),
                        max_cutoff,
                        table_dx,
                    )
            return self._analytic_kernel_hat_modes_from_table_python(
                indices,
                max_cutoff,
                table_dx,
            )
        if cpp_solvers is not None:
            func_name = (
                "analytic_kernel_hat_modes_miller64"
                if self.complex_dtype == np.dtype("complex64")
                else "analytic_kernel_hat_modes_miller"
            )
            return getattr(cpp_solvers, func_name)(
                np.ascontiguousarray(self.q_perp[indices], dtype=np.float64),
                np.ascontiguousarray(self.binned.r_centers, dtype=np.float64),
                int(self.binned.n_phi),
                max_cutoff,
                64,
            )

        modes = np.arange(max_cutoff + 1, dtype=int)
        x = self.q_perp[indices, None] * self.binned.r_centers[None, :]
        powers = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[
            np.mod(modes, 4)
        ]
        positive = (
            self.binned.n_phi
            * powers[None, None, :]
            * special.jv(modes[None, None, :], x[:, :, None])
        )
        return np.ascontiguousarray(positive, dtype=self.complex_dtype)

    def precompute_circular_kernels(self, q_block_size: int | None = None) -> None:
        """Precompute ``FFT_phi(exp(i q_perp R cos(phi)))`` for all q and R."""

        self._check_circular_grid()
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        kernel_hat = np.empty(
            (self.q.size, self.binned.r_centers.size, self.binned.n_phi),
            dtype=self.complex_dtype,
        )
        all_indices = np.arange(self.q.size)
        for start in range(0, self.q.size, block_size):
            indices = all_indices[start : start + block_size]
            phase = (
                self.q_perp[indices, None, None]
                * self.binned.r_centers[None, :, None]
                * self.cos_kernel_angles[None, None, :]
            )
            kernel = _unit_complex_from_phase_dtype(phase, self.complex_dtype)
            kernel_hat[indices] = scipy_fft.fft(kernel, axis=-1, workers=1)
        self._kernel_hat = kernel_hat

    def precompute_z_reduced(self, q_block_size: int | None = None) -> None:
        """Precompute ``sum_z exp(i q_z z) H[e,R,z,h]`` for all q."""

        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        z_reduced = np.empty(
            (
                self.q.size,
                self.hhat.shape[0],
                self.binned.r_centers.size,
                self.binned.n_phi,
            ),
            dtype=self.complex_dtype,
        )
        all_indices = np.arange(self.q.size)
        for start in range(0, self.q.size, block_size):
            indices = all_indices[start : start + block_size]
            z_reduced[indices] = np.einsum(
                "bz,erzh->berh",
                self.z_phase[indices],
                self.hhat,
                optimize=True,
            ).astype(self.complex_dtype, copy=False)
        self._z_reduced = z_reduced

    def _z_reduced_block(self, indices: np.ndarray) -> np.ndarray:
        if self._z_reduced is not None:
            return self._z_reduced[indices]
        return np.einsum(
            "bz,erzh->berh",
            self.z_phase[indices],
            self.hhat,
            optimize=True,
        ).astype(self.complex_dtype, copy=False)

    def _harmonic_indices_for_block(self, indices: np.ndarray) -> np.ndarray | None:
        if self.harmonic_bandlimit_margin is None:
            return None

        hmax = int(
            np.ceil(
                np.max(np.abs(self.q_perp[indices])) * self.binned.r_max
                + self.harmonic_bandlimit_margin
            )
        )
        if hmax >= self.n_half:
            return None
        if hmax <= 0:
            return np.array([0], dtype=np.intp)
        return np.r_[0 : hmax + 1, self.binned.n_phi - hmax : self.binned.n_phi].astype(
            np.intp,
            copy=False,
        )

    def _r_dependent_cutoff_matrix(
        self,
        indices: np.ndarray,
        *,
        margin: int,
        cutoff_bin_size: int,
    ) -> np.ndarray:
        cutoffs = np.ceil(
            np.abs(self.q_perp[indices, None]) * self.binned.r_centers[None, :]
            + margin
        ).astype(np.int64, copy=False)
        np.clip(cutoffs, 0, self.n_half, out=cutoffs)
        cutoffs = (
            ((cutoffs + cutoff_bin_size - 1) // cutoff_bin_size) * cutoff_bin_size
        )
        np.clip(cutoffs, 0, self.n_half, out=cutoffs)
        return np.ascontiguousarray(cutoffs, dtype=np.int64)

    def _sparse_rz_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._sparse_rz_cache is None:
            if self._sparse_only:
                raise RuntimeError(
                    "sparse RZ FFT requires a dense histogram; use a sparse-source solver"
                )
            active = np.nonzero(np.any(self.binned.hist != 0, axis=-1))
            if active[0].size == 0:
                hhat_active = np.empty((0, self.binned.n_phi), dtype=self.complex_dtype)
            else:
                hhat_active = np.ascontiguousarray(
                    self.hhat[active],
                    dtype=self.complex_dtype,
                )
            self._sparse_rz_cache = (
                np.ascontiguousarray(active[0], dtype=np.intp),
                np.ascontiguousarray(active[1], dtype=np.intp),
                np.ascontiguousarray(active[2], dtype=np.intp),
                hhat_active,
            )
        return self._sparse_rz_cache

    @property
    def active_rz_count(self) -> int:
        """Number of non-empty ``(element, R, z)`` beta profiles."""

        return int(self._sparse_rz_data()[0].size)

    def _sparse_flat_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._sparse_flat_cache is None:
            active = np.nonzero(self.binned.hist)
            if active[0].size == 0:
                values = np.empty(0, dtype=self.complex_dtype)
            else:
                values = np.ascontiguousarray(
                    self.binned.hist[active],
                    dtype=self.complex_dtype,
                )
            self._sparse_flat_cache = (
                np.ascontiguousarray(active[0], dtype=np.intp),
                np.ascontiguousarray(active[1], dtype=np.intp),
                np.ascontiguousarray(active[2], dtype=np.intp),
                np.ascontiguousarray(active[3], dtype=np.intp),
                values,
            )
        return self._sparse_flat_cache

    @property
    def active_flat_count(self) -> int:
        """Number of non-empty ``(element, R, z, beta)`` bins."""

        return int(self._sparse_flat_data()[0].size)

    def _sparse_profile_data(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self._sparse_profile_cache is None:
            active_e, active_r, active_z, active_beta, active_values = (
                self._sparse_flat_data()
            )
            if active_e.size == 0:
                starts = np.empty(0, dtype=np.intp)
                counts = np.empty(0, dtype=np.intp)
                profile_e = np.empty(0, dtype=np.intp)
                profile_r = np.empty(0, dtype=np.intp)
                profile_z = np.empty(0, dtype=np.intp)
            else:
                n_r = self.binned.r_centers.size
                n_z = self.binned.z_centers.size
                profile_ids = (
                    (active_e * n_r + active_r) * n_z + active_z
                )
                starts = np.r_[
                    0,
                    np.flatnonzero(np.diff(profile_ids)) + 1,
                ].astype(np.intp, copy=False)
                stops = np.r_[starts[1:], active_e.size].astype(np.intp, copy=False)
                counts = np.ascontiguousarray(stops - starts, dtype=np.intp)
                profile_e = np.ascontiguousarray(active_e[starts], dtype=np.intp)
                profile_r = np.ascontiguousarray(active_r[starts], dtype=np.intp)
                profile_z = np.ascontiguousarray(active_z[starts], dtype=np.intp)
                starts = np.ascontiguousarray(starts, dtype=np.intp)
            self._sparse_profile_cache = (
                profile_e,
                profile_r,
                profile_z,
                starts,
                counts,
                active_beta,
                active_values,
            )
        return self._sparse_profile_cache

    @property
    def active_sparse_profile_count(self) -> int:
        """Number of non-empty ``(element, R, z)`` profiles in sparse-flat form."""

        return int(self._sparse_profile_data()[0].size)

    def _sparse_er_profile_data(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self._sparse_er_profile_cache is None:
            active_e, active_r, active_z, active_beta, active_values = (
                self._sparse_flat_data()
            )
            if active_e.size == 0:
                starts = np.empty(0, dtype=np.intp)
                counts = np.empty(0, dtype=np.intp)
                profile_e = np.empty(0, dtype=np.intp)
                profile_r = np.empty(0, dtype=np.intp)
            else:
                n_r = self.binned.r_centers.size
                profile_ids = active_e * n_r + active_r
                starts = np.r_[
                    0,
                    np.flatnonzero(np.diff(profile_ids)) + 1,
                ].astype(np.intp, copy=False)
                stops = np.r_[starts[1:], active_e.size].astype(np.intp, copy=False)
                counts = np.ascontiguousarray(stops - starts, dtype=np.intp)
                profile_e = np.ascontiguousarray(active_e[starts], dtype=np.intp)
                profile_r = np.ascontiguousarray(active_r[starts], dtype=np.intp)
                starts = np.ascontiguousarray(starts, dtype=np.intp)
            self._sparse_er_profile_cache = (
                profile_e,
                profile_r,
                starts,
                counts,
                active_z,
                active_beta,
                active_values,
            )
        return self._sparse_er_profile_cache

    @property
    def active_er_profile_count(self) -> int:
        """Number of non-empty ``(element, R)`` sparse source-projection rows."""

        return int(self._sparse_er_profile_data()[0].size)

    def _sparse_twiddle(self, h_indices: np.ndarray) -> np.ndarray:
        h_indices = np.ascontiguousarray(h_indices, dtype=np.intp)
        key = tuple(int(h) for h in h_indices)
        cached = self._sparse_twiddle_cache.get(key)
        if cached is not None:
            return cached

        beta_indices = np.arange(self.binned.n_phi, dtype=float)
        phase = (
            -2.0
            * np.pi
            * beta_indices[:, None]
            * h_indices[None, :].astype(float)
            / float(self.binned.n_phi)
        )
        twiddle = _unit_complex_from_phase_dtype(phase, self.complex_dtype)
        self._sparse_twiddle_cache[key] = np.ascontiguousarray(
            twiddle,
            dtype=self.complex_dtype,
        )
        return self._sparse_twiddle_cache[key]

    def _sparse_profile_hhat(self, h_indices: np.ndarray) -> np.ndarray:
        h_indices = np.ascontiguousarray(h_indices, dtype=np.intp)
        key = tuple(int(h) for h in h_indices)
        cached = self._sparse_profile_hhat_cache.get(key)
        if cached is not None:
            return cached

        (
            _profile_e,
            _profile_r,
            _profile_z,
            profile_starts,
            profile_counts,
            active_beta,
            active_values,
        ) = self._sparse_profile_data()
        twiddle = self._sparse_twiddle(h_indices)
        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        if cpp_solvers is not None:
            build_name = (
                "build_sparse_profile_hhat64"
                if self.complex_dtype == np.dtype("complex64")
                else "build_sparse_profile_hhat"
            )
            hhat = getattr(cpp_solvers, build_name)(
                np.ascontiguousarray(profile_starts, dtype=np.int64),
                np.ascontiguousarray(profile_counts, dtype=np.int64),
                np.ascontiguousarray(active_beta, dtype=np.int64),
                np.ascontiguousarray(active_values, dtype=self.complex_dtype),
                twiddle,
            )
        else:
            hhat = np.empty(
                (profile_starts.size, h_indices.size),
                dtype=self.complex_dtype,
            )
            for p, (start, count) in enumerate(zip(profile_starts, profile_counts)):
                stop = start + count
                hhat[p] = np.sum(
                    active_values[start:stop, None] * twiddle[active_beta[start:stop]],
                    axis=0,
                    dtype=self.complex_dtype,
                )
        self._sparse_profile_hhat_cache[key] = np.ascontiguousarray(
            hhat,
            dtype=self.complex_dtype,
        )
        return self._sparse_profile_hhat_cache[key]

    def _adaptive_profile_hhat(
        self,
        h_indices: np.ndarray,
        *,
        dense_row_factor: float = 1.0,
        dense_batch_size: int = 2048,
    ) -> np.ndarray:
        dense_row_factor = float(dense_row_factor)
        if dense_row_factor < 0.0:
            raise ValueError("dense_row_factor must be non-negative")
        dense_batch_size = int(dense_batch_size)
        if dense_batch_size <= 0:
            raise ValueError("dense_batch_size must be positive")

        h_indices = np.ascontiguousarray(h_indices, dtype=np.intp)
        key = (tuple(int(h) for h in h_indices), dense_row_factor)
        cached = self._adaptive_profile_hhat_cache.get(key)
        if cached is not None:
            return cached

        (
            _profile_e,
            _profile_r,
            _profile_z,
            profile_starts,
            profile_counts,
            active_beta,
            active_values,
        ) = self._sparse_profile_data()
        n_profiles = int(profile_starts.size)
        n_h = int(h_indices.size)
        n_phi = int(self.binned.n_phi)
        dense_cost = dense_row_factor * n_phi * max(1.0, np.log2(max(n_phi, 2)))
        dense_mask = (profile_counts.astype(float) * max(n_h, 1)) >= dense_cost
        dense_profile_indices = np.flatnonzero(dense_mask)
        sparse_profile_indices = np.flatnonzero(~dense_mask)

        hhat = np.empty((n_profiles, n_h), dtype=self.complex_dtype)
        if sparse_profile_indices.size:
            twiddle = self._sparse_twiddle(h_indices)
            use_cpp = self.circular_backend in {"auto", "cpp"}
            cpp_solvers = (
                _cpp_solver_module(required=self.circular_backend == "cpp")
                if use_cpp
                else None
            )
            starts = np.ascontiguousarray(
                profile_starts[sparse_profile_indices],
                dtype=np.int64,
            )
            counts = np.ascontiguousarray(
                profile_counts[sparse_profile_indices],
                dtype=np.int64,
            )
            if cpp_solvers is not None:
                build_name = (
                    "build_sparse_profile_hhat64"
                    if self.complex_dtype == np.dtype("complex64")
                    else "build_sparse_profile_hhat"
                )
                sparse_hhat = getattr(cpp_solvers, build_name)(
                    starts,
                    counts,
                    np.ascontiguousarray(active_beta, dtype=np.int64),
                    np.ascontiguousarray(active_values, dtype=self.complex_dtype),
                    twiddle,
                )
            else:
                sparse_hhat = np.empty(
                    (sparse_profile_indices.size, n_h),
                    dtype=self.complex_dtype,
                )
                for out_i, p in enumerate(sparse_profile_indices):
                    start = profile_starts[p]
                    stop = start + profile_counts[p]
                    sparse_hhat[out_i] = np.sum(
                        active_values[start:stop, None] * twiddle[active_beta[start:stop]],
                        axis=0,
                        dtype=self.complex_dtype,
                    )
            hhat[sparse_profile_indices] = np.ascontiguousarray(
                sparse_hhat,
                dtype=self.complex_dtype,
            )

        if dense_profile_indices.size:
            for batch_start in range(0, dense_profile_indices.size, dense_batch_size):
                batch_profiles = dense_profile_indices[
                    batch_start : batch_start + dense_batch_size
                ]
                rows = np.zeros(
                    (batch_profiles.size, n_phi),
                    dtype=self.complex_dtype,
                )
                for row_i, p in enumerate(batch_profiles):
                    start = profile_starts[p]
                    stop = start + profile_counts[p]
                    rows[row_i, active_beta[start:stop]] = active_values[start:stop]
                rows_hhat = scipy_fft.fft(rows, axis=-1, workers=-1).astype(
                    self.complex_dtype,
                    copy=False,
                )
                hhat[batch_profiles] = np.ascontiguousarray(
                    np.take(rows_hhat, h_indices, axis=-1),
                    dtype=self.complex_dtype,
                )

        stats: dict[str, float | int] = {
            "n_profiles": n_profiles,
            "n_h": n_h,
            "n_phi": n_phi,
            "dense_row_factor": dense_row_factor,
            "dense_cost_threshold": float(dense_cost),
            "dense_profile_count": int(dense_profile_indices.size),
            "sparse_profile_count": int(sparse_profile_indices.size),
            "dense_profile_fraction": (
                float(dense_profile_indices.size / n_profiles)
                if n_profiles
                else 0.0
            ),
            "sparse_profile_fraction": (
                float(sparse_profile_indices.size / n_profiles)
                if n_profiles
                else 0.0
            ),
            "dense_active_flat_bins": int(np.sum(profile_counts[dense_profile_indices]))
            if dense_profile_indices.size
            else 0,
            "sparse_active_flat_bins": int(np.sum(profile_counts[sparse_profile_indices]))
            if sparse_profile_indices.size
            else 0,
        }
        self._adaptive_profile_hhat_cache[key] = np.ascontiguousarray(
            hhat,
            dtype=self.complex_dtype,
        )
        self._adaptive_profile_hhat_stats[key] = stats
        return self._adaptive_profile_hhat_cache[key]

    @property
    def last_adaptive_profile_stats(self) -> dict[str, float | int] | None:
        return self._last_adaptive_profile_stats

    def _circular_ahat(
        self,
        indices: np.ndarray,
        *,
        form_factors: np.ndarray | None = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")
        ff = self.form_factors if form_factors is None else form_factors
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        out = np.empty((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        use_cpp = self.circular_backend == "cpp" or (
            self.circular_backend == "auto" and self._z_reduced is not None
        )
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is not None:
                khat = np.take(khat, hsel, axis=-1)
            if cpp_solvers is not None:
                ff_block = np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype)
                khat = np.ascontiguousarray(khat, dtype=self.complex_dtype)
                fused_name = (
                    "circular_contract_fused64"
                    if self.complex_dtype == np.dtype("complex64")
                    else "circular_contract_fused"
                )
                z_reduced_name = (
                    "circular_contract_z_reduced64"
                    if self.complex_dtype == np.dtype("complex64")
                    else "circular_contract_z_reduced"
                )
                if self._z_reduced is None:
                    use_block_z = self.complex_dtype == np.dtype("complex64") and (
                        hsel is not None or self.binned.n_phi <= 256
                    )
                    if use_block_z:
                        hhat_block = (
                            self.hhat
                            if hsel is None
                            else np.take(self.hhat, hsel, axis=-1)
                        )
                        z_reduced = np.einsum(
                            "bz,erzh->berh",
                            self.z_phase[sel],
                            hhat_block,
                            optimize=True,
                        ).astype(self.complex_dtype, copy=False)
                        ahat_block = getattr(cpp_solvers, z_reduced_name)(
                            np.ascontiguousarray(
                                z_reduced,
                                dtype=self.complex_dtype,
                            ),
                            khat,
                            ff_block,
                        )
                    else:
                        hhat_cpp = np.ascontiguousarray(
                            self.hhat
                            if hsel is None
                            else np.take(self.hhat, hsel, axis=-1),
                            dtype=self.complex_dtype,
                        )
                        ahat_block = getattr(cpp_solvers, fused_name)(
                            hhat_cpp,
                            np.ascontiguousarray(
                                self.z_phase[sel],
                                dtype=self.complex_dtype,
                            ),
                            khat,
                            ff_block,
                        )
                else:
                    z_reduced = self._z_reduced_block(sel)
                    if hsel is not None:
                        z_reduced = np.take(z_reduced, hsel, axis=-1)
                    ahat_block = getattr(cpp_solvers, z_reduced_name)(
                        np.ascontiguousarray(z_reduced, dtype=self.complex_dtype),
                        khat,
                        ff_block,
                    )
                if hsel is None:
                    out[local] = ahat_block
                else:
                    out[local] = 0.0
                    out[local, hsel] = ahat_block
                continue

            if hsel is None:
                z_reduced = self._z_reduced_block(sel)
            elif self._z_reduced is not None:
                z_reduced = np.take(self._z_reduced_block(sel), hsel, axis=-1)
            else:
                z_reduced = np.einsum(
                    "bz,erzh->berh",
                    self.z_phase[sel],
                    np.take(self.hhat, hsel, axis=-1),
                    optimize=True,
                )
            if z_reduced.shape[1] == 1:
                ahat_block = ff[0, sel, None] * np.einsum(
                    "brh,brh->bh",
                    z_reduced[:, 0],
                    khat,
                    optimize=True,
                )
            else:
                ahat_block = np.einsum(
                    "eb,berh,brh->bh",
                    ff[:, sel],
                    z_reduced,
                    khat,
                    optimize=True,
                )
            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block
        return out

    def circular_fft(
        self,
        q_indices: np.ndarray | None = None,
        *,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate the circular FFT solver using cached grid tables."""

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ahat = self._circular_ahat(indices, q_block_size=block_size)
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_r_dependent_bandlimit(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        analytic_kernel_table_dx: float | None = None,
        z_projection: bool = False,
        r_block_size: int | None = None,
        fused_analytic_kernel: bool = False,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return cake-map Fourier coefficients with R-dependent harmonic cutoffs."""

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        margin = int(margin)
        if margin < 0:
            raise ValueError("margin must be non-negative")
        cutoff_bin_size = int(cutoff_bin_size)
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")
        if analytic_kernel_table_dx is None:
            analytic_kernel_table_dx = self.analytic_kernel_table_dx
        if analytic_kernel_table_dx is not None and analytic_kernel_table_dx <= 0:
            raise ValueError("analytic_kernel_table_dx must be positive")
        if fused_analytic_kernel and not analytic_kernel:
            raise ValueError("fused_analytic_kernel requires analytic_kernel=True")
        if fused_analytic_kernel and analytic_kernel_table_dx is not None:
            raise ValueError("fused_analytic_kernel does not support table interpolation")
        if r_block_size is not None:
            r_block_size = int(r_block_size)
            if r_block_size <= 0:
                raise ValueError("r_block_size must be positive")
        if z_projection and r_block_size is not None:
            raise ValueError("r_block_size is not supported with z_projection")
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        if cpp_solvers is None and fused_analytic_kernel:
            raise ValueError("fused_analytic_kernel requires the C++ circular backend")
        if cpp_solvers is None and r_block_size is not None:
            raise ValueError("r_block_size requires the C++ circular backend")
        out = np.empty((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if cpp_solvers is not None:
            sampled_func_name = (
                "circular_contract_r_dependent64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_contract_r_dependent"
            )
            compact_func_name = (
                "circular_contract_r_dependent_modes64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_contract_r_dependent_modes"
            )
            compact_half_func_name = (
                "circular_contract_r_dependent_half_modes64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_contract_r_dependent_half_modes"
            )
            compact_half_z_func_name = (
                "circular_contract_r_dependent_half_z_reduced64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_contract_r_dependent_half_z_reduced"
            )
            sampled_func = getattr(cpp_solvers, sampled_func_name)
            compact_func = getattr(cpp_solvers, compact_func_name)
            compact_half_func = getattr(cpp_solvers, compact_half_func_name, None)
            compact_half_z_func = getattr(cpp_solvers, compact_half_z_func_name, None)
            fused_half_func_name = (
                "circular_contract_r_dependent_half_modes_miller64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_contract_r_dependent_half_modes_miller"
            )
            fused_half_func = getattr(cpp_solvers, fused_half_func_name, None)
            if fused_analytic_kernel and fused_half_func is None:
                raise ValueError("C++ fused Miller R-dependent contraction is unavailable")
            hhat = None
            hhat_positive = None
            hhat_half_by_modes: dict[int, np.ndarray] = {}
            for start in range(0, indices.size, block_size):
                local = slice(start, min(start + block_size, indices.size))
                sel = indices[local]
                cutoffs = self._r_dependent_cutoff_matrix(
                    sel,
                    margin=margin,
                    cutoff_bin_size=cutoff_bin_size,
                )
                max_cutoff = int(np.max(cutoffs)) if cutoffs.size else 0
                z_block = np.ascontiguousarray(
                    self.z_phase[sel],
                    dtype=self.complex_dtype,
                )
                ff_block = np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype)
                if (
                    analytic_kernel
                    and max_cutoff < self.n_half
                    and self.has_real_histogram
                    and r_block_size is not None
                    and compact_half_func is not None
                ):
                    block_out = np.zeros(
                        (sel.size, self.binned.n_phi),
                        dtype=self.complex_dtype,
                    )
                    n_r = self.binned.r_centers.size
                    for r_start in range(0, n_r, r_block_size):
                        r_stop = min(r_start + r_block_size, n_r)
                        r_idx = np.arange(r_start, r_stop, dtype=np.intp)
                        hhat_block = scipy_fft.rfft(
                            np.ascontiguousarray(
                                self.binned.hist[:, r_start:r_stop, :, :],
                            ),
                            axis=-1,
                            workers=1,
                        ).astype(self.complex_dtype, copy=False)
                        cutoffs_block = np.ascontiguousarray(
                            cutoffs[:, r_start:r_stop],
                            dtype=np.int64,
                        )
                        if fused_analytic_kernel and fused_half_func is not None:
                            block_out += fused_half_func(
                                np.ascontiguousarray(
                                    hhat_block,
                                    dtype=self.complex_dtype,
                                ),
                                z_block,
                                np.ascontiguousarray(
                                    self.q_perp[sel],
                                    dtype=np.float64,
                                ),
                                np.ascontiguousarray(
                                    self.binned.r_centers[r_start:r_stop],
                                    dtype=np.float64,
                                ),
                                ff_block,
                                cutoffs_block,
                                int(self.binned.n_phi),
                                max_cutoff,
                                64,
                            )
                        else:
                            block_out += compact_half_func(
                                np.ascontiguousarray(
                                    hhat_block,
                                    dtype=self.complex_dtype,
                                ),
                                z_block,
                                self._analytic_kernel_hat_modes_r(
                                    sel,
                                    r_idx,
                                    max_cutoff,
                                    table_dx=analytic_kernel_table_dx,
                                ),
                                ff_block,
                                cutoffs_block,
                                int(self.binned.n_phi),
                                max_cutoff,
                            )
                    out[local] = block_out
                elif (
                    analytic_kernel
                    and max_cutoff < self.n_half
                    and self.has_real_histogram
                    and fused_analytic_kernel
                    and fused_half_func is not None
                ):
                    if hhat_positive is None:
                        hhat_positive = self._hhat_positive_modes_for_half_contraction()
                    out[local] = fused_half_func(
                        hhat_positive,
                        z_block,
                        np.ascontiguousarray(self.q_perp[sel], dtype=np.float64),
                        np.ascontiguousarray(
                            self.binned.r_centers,
                            dtype=np.float64,
                        ),
                        ff_block,
                        cutoffs,
                        int(self.binned.n_phi),
                        max_cutoff,
                        64,
                    )
                elif (
                    analytic_kernel
                    and max_cutoff < self.n_half
                    and self.has_real_histogram
                    and z_projection
                    and compact_half_z_func is not None
                ):
                    hhat_modes = hhat_half_by_modes.get(max_cutoff + 1)
                    if hhat_modes is None:
                        hhat_modes = self.hhat_half_modes(max_cutoff)
                        hhat_half_by_modes[max_cutoff + 1] = hhat_modes
                    z_pos = np.einsum(
                        "bz,erzh->berh",
                        z_block,
                        hhat_modes,
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)
                    z_neg = np.einsum(
                        "bz,erzh->berh",
                        z_block,
                        np.conj(hhat_modes),
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)
                    out[local] = compact_half_z_func(
                        np.ascontiguousarray(z_pos, dtype=self.complex_dtype),
                        np.ascontiguousarray(z_neg, dtype=self.complex_dtype),
                        self._analytic_kernel_hat_modes(
                            sel,
                            max_cutoff,
                            table_dx=analytic_kernel_table_dx,
                        ),
                        ff_block,
                        cutoffs,
                        int(self.binned.n_phi),
                        max_cutoff,
                    )
                elif (
                    analytic_kernel
                    and max_cutoff < self.n_half
                    and self.has_real_histogram
                    and compact_half_func is not None
                ):
                    if hhat_positive is None:
                        hhat_positive = self._hhat_positive_modes_for_half_contraction()
                    out[local] = compact_half_func(
                        hhat_positive,
                        z_block,
                        self._analytic_kernel_hat_modes(
                            sel,
                            max_cutoff,
                            table_dx=analytic_kernel_table_dx,
                        ),
                        ff_block,
                        cutoffs,
                        int(self.binned.n_phi),
                        max_cutoff,
                    )
                elif analytic_kernel and max_cutoff < self.n_half:
                    if hhat is None:
                        hhat = np.ascontiguousarray(
                            self.hhat,
                            dtype=self.complex_dtype,
                        )
                    out[local] = compact_func(
                        hhat,
                        z_block,
                        self._analytic_kernel_hat_modes(
                            sel,
                            max_cutoff,
                            table_dx=analytic_kernel_table_dx,
                        ),
                        ff_block,
                        cutoffs,
                        max_cutoff,
                    )
                else:
                    if hhat is None:
                        hhat = np.ascontiguousarray(
                            self.hhat,
                            dtype=self.complex_dtype,
                        )
                    out[local] = sampled_func(
                        hhat,
                        z_block,
                        np.ascontiguousarray(
                            self._kernel_hat_block(sel),
                            dtype=self.complex_dtype,
                        ),
                        ff_block,
                        cutoffs,
                    )
            return out

        modes = np.fft.fftfreq(self.binned.n_phi, d=1.0 / self.binned.n_phi).astype(
            int
        )
        abs_modes = np.abs(modes)

        for out_i, iq in enumerate(indices):
            khat = self._kernel_hat_block(np.array([iq], dtype=np.intp))[0]
            cutoffs = self._r_dependent_cutoff_matrix(
                np.array([iq], dtype=np.intp),
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
            )[0]
            ahat = np.zeros(self.binned.n_phi, dtype=self.complex_dtype)

            for cutoff in np.unique(cutoffs):
                r_idx = np.flatnonzero(cutoffs == cutoff)
                if r_idx.size == 0:
                    continue
                if cutoff >= self.n_half:
                    hsel = np.arange(self.binned.n_phi, dtype=np.intp)
                else:
                    hsel = np.flatnonzero(abs_modes <= cutoff).astype(
                        np.intp,
                        copy=False,
                    )
                if hsel.size == 0:
                    continue

                khat_rh = np.take(np.take(khat, r_idx, axis=0), hsel, axis=-1)
                if self.hhat.shape[0] == 1:
                    hhat_r = np.take(
                        np.take(self.hhat[0], r_idx, axis=0),
                        hsel,
                        axis=-1,
                    )
                    b_rh = np.einsum(
                        "z,rzh->rh",
                        self.z_phase[iq],
                        hhat_r,
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)
                    b_rh *= ff[0, iq]
                else:
                    hhat_r = np.take(
                        np.take(self.hhat, r_idx, axis=1),
                        hsel,
                        axis=-1,
                    )
                    z_reduced = np.einsum(
                        "z,erzh->erh",
                        self.z_phase[iq],
                        hhat_r,
                        optimize=True,
                    )
                    b_rh = np.einsum(
                        "e,erh->rh",
                        ff[:, iq],
                        z_reduced,
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)
                ahat[hsel] += np.einsum("rh,rh->h", b_rh, khat_rh, optimize=True)
            out[out_i] = ahat
        return out

    def circular_fft_r_dependent_bandlimit(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        analytic_kernel_table_dx: float | None = None,
        z_projection: bool = False,
        r_block_size: int | None = None,
        fused_analytic_kernel: bool = False,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate a full cake map with R-dependent harmonic cutoffs."""

        ahat = self.circular_ahat_r_dependent_bandlimit(
            q_indices=q_indices,
            form_factors=form_factors,
            margin=margin,
            cutoff_bin_size=cutoff_bin_size,
            analytic_kernel=analytic_kernel,
            analytic_kernel_table_dx=analytic_kernel_table_dx,
            z_projection=z_projection,
            r_block_size=r_block_size,
            fused_analytic_kernel=fused_analytic_kernel,
            q_block_size=q_block_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_sparse_rz(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        active_chunk_size: int = 256,
    ) -> np.ndarray:
        """Return Fourier-domain ring coefficients using active ``R/z`` cells only."""

        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")
        active_chunk_size = int(active_chunk_size)
        if active_chunk_size <= 0:
            raise ValueError("active_chunk_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        active_e, active_r, active_z, active_hhat = self._sparse_rz_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if active_e.size == 0:
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        sparse_name = (
            "circular_contract_sparse_rz64"
            if self.complex_dtype == np.dtype("complex64")
            else "circular_contract_sparse_rz"
        )

        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is not None:
                khat = np.take(khat, hsel, axis=-1)
            khat = np.ascontiguousarray(khat, dtype=self.complex_dtype)

            if cpp_solvers is not None:
                hhat_cpp = active_hhat if hsel is None else np.take(active_hhat, hsel, axis=-1)
                ahat_block = getattr(cpp_solvers, sparse_name)(
                    np.ascontiguousarray(active_e, dtype=np.int64),
                    np.ascontiguousarray(active_r, dtype=np.int64),
                    np.ascontiguousarray(active_z, dtype=np.int64),
                    np.ascontiguousarray(hhat_cpp, dtype=self.complex_dtype),
                    np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                    khat,
                    np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
                )
                if hsel is None:
                    out[local] = ahat_block
                else:
                    out[local] = 0.0
                    out[local, hsel] = ahat_block
                continue

            block_out = np.zeros((sel.size, khat.shape[-1]), dtype=self.complex_dtype)
            for chunk_start in range(0, active_e.size, active_chunk_size):
                chunk = slice(
                    chunk_start,
                    min(chunk_start + active_chunk_size, active_e.size),
                )
                e_chunk = active_e[chunk]
                r_chunk = active_r[chunk]
                z_chunk = active_z[chunk]
                h_chunk = active_hhat[chunk]
                if hsel is not None:
                    h_chunk = np.take(h_chunk, hsel, axis=-1)

                coeff = (
                    self.z_phase[sel[:, None], z_chunk[None, :]]
                    * ff[e_chunk[None, :], sel[:, None]]
                )
                khat_chunk = np.take(khat, r_chunk, axis=1)
                block_out += np.einsum(
                    "bc,bch,ch->bh",
                    coeff,
                    khat_chunk,
                    h_chunk,
                    optimize=True,
                )

            if hsel is None:
                out[local] = block_out
            else:
                out[local] = 0.0
                out[local, hsel] = block_out
        return out

    def circular_fft_sparse_rz(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        active_chunk_size: int = 256,
    ) -> np.ndarray:
        """Evaluate the circular solver by contracting only active ``R/z`` cells."""

        self._check_circular_grid()
        ahat = self.circular_ahat_sparse_rz(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
            active_chunk_size=active_chunk_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_sparse_flat(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        active_chunk_size: int = 256,
    ) -> np.ndarray:
        """Return Fourier-domain ring coefficients from active flat bins only.

        This path exploits ``(element, R, z, beta)`` sparsity. It is exact
        relative to the dense binned circular solver, but its cost scales as
        ``active_flat_bins * active_harmonics * n_q`` and is therefore useful
        only when angular profiles are very sparse.
        """

        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")
        active_chunk_size = int(active_chunk_size)
        if active_chunk_size <= 0:
            raise ValueError("active_chunk_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        active_e, active_r, active_z, active_beta, active_values = self._sparse_flat_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if active_e.size == 0:
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        sparse_name = (
            "circular_contract_sparse_flat64"
            if self.complex_dtype == np.dtype("complex64")
            else "circular_contract_sparse_flat"
        )

        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is None:
                h_indices = np.arange(self.binned.n_phi, dtype=np.intp)
            else:
                h_indices = hsel
                khat = np.take(khat, hsel, axis=-1)
            khat = np.ascontiguousarray(khat, dtype=self.complex_dtype)
            twiddle = self._sparse_twiddle(h_indices)

            if cpp_solvers is not None:
                ahat_block = getattr(cpp_solvers, sparse_name)(
                    np.ascontiguousarray(active_e, dtype=np.int64),
                    np.ascontiguousarray(active_r, dtype=np.int64),
                    np.ascontiguousarray(active_z, dtype=np.int64),
                    np.ascontiguousarray(active_beta, dtype=np.int64),
                    np.ascontiguousarray(active_values, dtype=self.complex_dtype),
                    twiddle,
                    np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                    khat,
                    np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
                )
            else:
                ahat_block = np.zeros((sel.size, h_indices.size), dtype=self.complex_dtype)
                for chunk_start in range(0, active_e.size, active_chunk_size):
                    chunk = slice(
                        chunk_start,
                        min(chunk_start + active_chunk_size, active_e.size),
                    )
                    e_chunk = active_e[chunk]
                    r_chunk = active_r[chunk]
                    z_chunk = active_z[chunk]
                    beta_chunk = active_beta[chunk]
                    value_chunk = active_values[chunk]
                    coeff = (
                        self.z_phase[sel[:, None], z_chunk[None, :]]
                        * ff[e_chunk[None, :], sel[:, None]]
                        * value_chunk[None, :]
                    )
                    khat_chunk = np.take(khat, r_chunk, axis=1)
                    twiddle_chunk = np.take(twiddle, beta_chunk, axis=0)
                    ahat_block += np.einsum(
                        "bc,bch,ch->bh",
                        coeff,
                        khat_chunk,
                        twiddle_chunk,
                        optimize=True,
                    )

            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block
        return out

    def circular_fft_sparse_flat(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        active_chunk_size: int = 256,
    ) -> np.ndarray:
        """Evaluate the circular solver by contracting active flat bins only."""

        self._check_circular_grid()
        ahat = self.circular_ahat_sparse_flat(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
            active_chunk_size=active_chunk_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_sparse_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return Fourier-domain ring coefficients from grouped sparse profiles.

        This is a sparse-flat path grouped by ``(element, R, z)``. It avoids the
        dense angular FFT and reuses the same axial, form-factor, and radial
        terms for all non-empty beta bins in a profile.
        """

        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        (
            profile_e,
            profile_r,
            profile_z,
            _profile_starts,
            _profile_counts,
            _active_beta,
            _active_values,
        ) = self._sparse_profile_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if profile_e.size == 0:
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        sparse_name = (
            "circular_contract_sparse_rz64"
            if self.complex_dtype == np.dtype("complex64")
            else "circular_contract_sparse_rz"
        )

        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is None:
                h_indices = np.arange(self.binned.n_phi, dtype=np.intp)
            else:
                h_indices = hsel
                khat = np.take(khat, hsel, axis=-1)
            khat = np.ascontiguousarray(khat, dtype=self.complex_dtype)
            active_hhat = self._sparse_profile_hhat(h_indices)

            if cpp_solvers is not None:
                ahat_block = getattr(cpp_solvers, sparse_name)(
                    np.ascontiguousarray(profile_e, dtype=np.int64),
                    np.ascontiguousarray(profile_r, dtype=np.int64),
                    np.ascontiguousarray(profile_z, dtype=np.int64),
                    np.ascontiguousarray(active_hhat, dtype=self.complex_dtype),
                    np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                    khat,
                    np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
                )
            else:
                ahat_block = np.zeros((sel.size, h_indices.size), dtype=self.complex_dtype)
                for chunk_start in range(0, profile_e.size, 256):
                    chunk = slice(chunk_start, min(chunk_start + 256, profile_e.size))
                    e_chunk = profile_e[chunk]
                    r_chunk = profile_r[chunk]
                    z_chunk = profile_z[chunk]
                    h_chunk = active_hhat[chunk]
                    coeff = (
                        self.z_phase[sel[:, None], z_chunk[None, :]]
                        * ff[e_chunk[None, :], sel[:, None]]
                    )
                    khat_chunk = np.take(khat, r_chunk, axis=1)
                    ahat_block += (
                        np.einsum(
                            "bc,bch,ch->bh",
                            coeff,
                            khat_chunk,
                            h_chunk,
                            optimize=True,
                        )
                    )

            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block
        return out

    def circular_fft_sparse_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate the circular solver with grouped sparse angular profiles."""

        self._check_circular_grid()
        ahat = self.circular_ahat_sparse_profiles(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_sparse_source_projection(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        profile_chunk_size: int | None = 64,
    ) -> np.ndarray:
        """Return Fourier coefficients using sparse ``z,beta`` source projection.

        For each q block this path first accumulates active
        ``(element, R, z, beta)`` bins into complex ``(q, element, R, beta)``
        source rows, then FFTs the beta axis once per active ``(element, R)``
        row. It is exact relative to the same dense binned circular solver, and
        is intended for fine high-q grids where the dense z reduction is mostly
        empty.
        """

        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")
        if profile_chunk_size is not None:
            profile_chunk_size = int(profile_chunk_size)
            if profile_chunk_size <= 0:
                raise ValueError("profile_chunk_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        (
            profile_e,
            profile_r,
            profile_starts,
            profile_counts,
            active_z,
            active_beta,
            active_values,
        ) = self._sparse_er_profile_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if profile_e.size == 0:
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        build_projection = None
        if cpp_solvers is not None:
            build_name = (
                "build_sparse_source_projection64"
                if self.complex_dtype == np.dtype("complex64")
                else "build_sparse_source_projection"
            )
            if self.circular_backend == "cpp":
                build_projection = getattr(cpp_solvers, build_name)
            else:
                build_projection = getattr(cpp_solvers, build_name, None)
        active_z_cpp = None
        active_beta_cpp = None
        active_values_cpp = None
        if build_projection is not None:
            active_z_cpp = np.ascontiguousarray(active_z, dtype=np.int64)
            active_beta_cpp = np.ascontiguousarray(active_beta, dtype=np.int64)
            active_values_cpp = np.ascontiguousarray(
                active_values,
                dtype=self.complex_dtype,
            )

        chunk_size = profile_e.size if profile_chunk_size is None else profile_chunk_size
        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            z_phase_block = self.z_phase[sel]
            hsel = self._harmonic_indices_for_block(sel)
            n_h = self.binned.n_phi if hsel is None else hsel.size

            ahat_block = np.zeros(
                (sel.size, n_h),
                dtype=self.complex_dtype,
            )
            q_rows = np.arange(sel.size, dtype=np.intp)[:, None]
            for profile_start in range(0, profile_e.size, chunk_size):
                profile_stop = min(profile_start + chunk_size, profile_e.size)
                profile_slice = slice(profile_start, profile_stop)
                profile_count = profile_stop - profile_start
                if build_projection is not None:
                    projected = build_projection(
                        np.ascontiguousarray(
                            profile_starts[profile_slice],
                            dtype=np.int64,
                        ),
                        np.ascontiguousarray(
                            profile_counts[profile_slice],
                            dtype=np.int64,
                        ),
                        active_z_cpp,
                        active_beta_cpp,
                        active_values_cpp,
                        np.ascontiguousarray(z_phase_block, dtype=self.complex_dtype),
                        int(self.binned.n_phi),
                    )
                else:
                    projected = np.zeros(
                        (sel.size, profile_count, self.binned.n_phi),
                        dtype=self.complex_dtype,
                    )

                    for local_profile, global_profile in enumerate(
                        range(profile_start, profile_stop)
                    ):
                        flat_start = int(profile_starts[global_profile])
                        flat_stop = flat_start + int(profile_counts[global_profile])
                        z_idx = active_z[flat_start:flat_stop]
                        beta_idx = active_beta[flat_start:flat_stop]
                        values = active_values[flat_start:flat_stop]
                        contribution = z_phase_block[:, z_idx] * values[None, :]
                        np.add.at(
                            projected[:, local_profile, :],
                            (q_rows, beta_idx[None, :]),
                            contribution,
                        )

                projected_hhat = scipy_fft.fft(
                    projected,
                    axis=-1,
                    workers=1,
                    overwrite_x=True,
                ).astype(self.complex_dtype, copy=False)
                if hsel is not None:
                    projected_hhat = np.take(projected_hhat, hsel, axis=-1)

                profile_e_chunk = profile_e[profile_slice]
                profile_r_chunk = profile_r[profile_slice]
                ff_profiles = ff[profile_e_chunk][:, sel].T
                unique_r, inverse_r = np.unique(profile_r_chunk, return_inverse=True)
                khat_unique = self._kernel_hat_block_r(sel, unique_r)
                if hsel is not None:
                    khat_unique = np.take(khat_unique, hsel, axis=-1)
                khat_profiles = np.take(khat_unique, inverse_r, axis=1)
                ahat_block += np.einsum(
                    "bp,bph,bph->bh",
                    ff_profiles,
                    projected_hhat,
                    khat_profiles,
                    optimize=True,
                )

            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block
        return out

    def circular_fft_sparse_source_projection(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        profile_chunk_size: int | None = 64,
    ) -> np.ndarray:
        """Evaluate circular FFT with sparse q-dependent source projection."""

        self._check_circular_grid()
        ahat = self.circular_ahat_sparse_source_projection(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
            profile_chunk_size=profile_chunk_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_sparse_source_r_dependent(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        analytic_kernel_table_dx: float | None = None,
        q_block_size: int | None = None,
        profile_chunk_size: int | None = 64,
    ) -> np.ndarray:
        """Return Fourier coefficients with sparse source projection and R cutoffs.

        This combines the q-dependent sparse ``(z, beta)`` source projection
        with the same R-dependent harmonic cutoffs used by
        :meth:`circular_ahat_r_dependent_bandlimit`. It is therefore exact
        relative to that R-dependent sampled-kernel path, up to floating-point
        accumulation order.
        """

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        margin = int(margin)
        if margin < 0:
            raise ValueError("margin must be non-negative")
        cutoff_bin_size = int(cutoff_bin_size)
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")
        if analytic_kernel_table_dx is None:
            analytic_kernel_table_dx = self.analytic_kernel_table_dx
        if analytic_kernel_table_dx is not None and analytic_kernel_table_dx <= 0:
            raise ValueError("analytic_kernel_table_dx must be positive")
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")
        if profile_chunk_size is not None:
            profile_chunk_size = int(profile_chunk_size)
            if profile_chunk_size <= 0:
                raise ValueError("profile_chunk_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        (
            profile_e,
            profile_r,
            profile_starts,
            profile_counts,
            active_z,
            active_beta,
            active_values,
        ) = self._sparse_er_profile_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if profile_e.size == 0:
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        build_projection = None
        if cpp_solvers is not None:
            build_name = (
                "build_sparse_source_projection64"
                if self.complex_dtype == np.dtype("complex64")
                else "build_sparse_source_projection"
            )
            if self.circular_backend == "cpp":
                build_projection = getattr(cpp_solvers, build_name)
            else:
                build_projection = getattr(cpp_solvers, build_name, None)
        contract_r_dependent = None
        if cpp_solvers is not None:
            contract_name = (
                "sparse_source_r_dependent_contract64"
                if self.complex_dtype == np.dtype("complex64")
                else "sparse_source_r_dependent_contract"
            )
            contract_r_dependent = getattr(cpp_solvers, contract_name, None)
        active_z_cpp = None
        active_beta_cpp = None
        active_values_cpp = None
        if build_projection is not None:
            active_z_cpp = np.ascontiguousarray(active_z, dtype=np.int64)
            active_beta_cpp = np.ascontiguousarray(active_beta, dtype=np.int64)
            active_values_cpp = np.ascontiguousarray(
                active_values,
                dtype=self.complex_dtype,
            )

        modes = np.fft.fftfreq(self.binned.n_phi, d=1.0 / self.binned.n_phi).astype(
            int
        )
        abs_modes = np.abs(modes)
        chunk_size = profile_e.size if profile_chunk_size is None else profile_chunk_size
        q_rows = None
        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            z_phase_block = self.z_phase[sel]
            cutoffs = self._r_dependent_cutoff_matrix(
                sel,
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
            )
            max_cutoff = int(np.max(cutoffs)) if cutoffs.size else 0
            if max_cutoff >= self.n_half:
                hsel = None
                h_indices = np.arange(self.binned.n_phi, dtype=np.intp)
                h_abs = abs_modes
            else:
                hsel = np.flatnonzero(abs_modes <= max_cutoff).astype(
                    np.intp,
                    copy=False,
                )
                h_indices = hsel
                h_abs = abs_modes[hsel]

            ahat_block = np.zeros(
                (sel.size, h_indices.size),
                dtype=self.complex_dtype,
            )
            if q_rows is None or q_rows.shape[0] != sel.size:
                q_rows = np.arange(sel.size, dtype=np.intp)[:, None]
            for profile_start in range(0, profile_e.size, chunk_size):
                profile_stop = min(profile_start + chunk_size, profile_e.size)
                profile_slice = slice(profile_start, profile_stop)
                profile_count = profile_stop - profile_start
                if build_projection is not None:
                    projected = build_projection(
                        np.ascontiguousarray(
                            profile_starts[profile_slice],
                            dtype=np.int64,
                        ),
                        np.ascontiguousarray(
                            profile_counts[profile_slice],
                            dtype=np.int64,
                        ),
                        active_z_cpp,
                        active_beta_cpp,
                        active_values_cpp,
                        np.ascontiguousarray(z_phase_block, dtype=self.complex_dtype),
                        int(self.binned.n_phi),
                    )
                else:
                    projected = np.zeros(
                        (sel.size, profile_count, self.binned.n_phi),
                        dtype=self.complex_dtype,
                    )
                    for local_profile, global_profile in enumerate(
                        range(profile_start, profile_stop)
                    ):
                        flat_start = int(profile_starts[global_profile])
                        flat_stop = flat_start + int(profile_counts[global_profile])
                        z_idx = active_z[flat_start:flat_stop]
                        beta_idx = active_beta[flat_start:flat_stop]
                        values = active_values[flat_start:flat_stop]
                        contribution = z_phase_block[:, z_idx] * values[None, :]
                        np.add.at(
                            projected[:, local_profile, :],
                            (q_rows, beta_idx[None, :]),
                            contribution,
                        )

                projected_hhat = scipy_fft.fft(
                    projected,
                    axis=-1,
                    workers=1,
                    overwrite_x=True,
                ).astype(self.complex_dtype, copy=False)
                if hsel is not None:
                    projected_hhat = np.take(projected_hhat, hsel, axis=-1)

                profile_e_chunk = profile_e[profile_slice]
                profile_r_chunk = profile_r[profile_slice]
                ff_profiles = ff[profile_e_chunk][:, sel].T
                unique_r, inverse_r = np.unique(profile_r_chunk, return_inverse=True)
                if analytic_kernel and max_cutoff < self.n_half:
                    khat_positive = self._analytic_kernel_hat_modes_r(
                        sel,
                        unique_r,
                        max_cutoff,
                        table_dx=analytic_kernel_table_dx,
                    )
                    khat_unique = np.take(khat_positive, h_abs, axis=-1)
                else:
                    khat_unique = self._kernel_hat_block_r(sel, unique_r)
                    if hsel is not None:
                        khat_unique = np.take(khat_unique, hsel, axis=-1)
                cutoff_profiles = np.take(cutoffs, profile_r_chunk, axis=1)
                if contract_r_dependent is not None:
                    ahat_block += contract_r_dependent(
                        np.ascontiguousarray(
                            projected_hhat,
                            dtype=self.complex_dtype,
                        ),
                        np.ascontiguousarray(khat_unique, dtype=self.complex_dtype),
                        np.ascontiguousarray(ff_profiles, dtype=self.complex_dtype),
                        np.ascontiguousarray(cutoff_profiles, dtype=np.int64),
                        np.ascontiguousarray(inverse_r, dtype=np.int64),
                        np.ascontiguousarray(h_abs, dtype=np.int64),
                    )
                else:
                    khat_profiles = np.take(khat_unique, inverse_r, axis=1)
                    for q_row in range(sel.size):
                        cutoffs_row = cutoff_profiles[q_row]
                        for cutoff in np.unique(cutoffs_row):
                            p_pos = np.flatnonzero(cutoffs_row == cutoff)
                            h_pos = np.flatnonzero(h_abs <= cutoff)
                            if p_pos.size == 0 or h_pos.size == 0:
                                continue
                            projected_sel = projected_hhat[q_row][
                                np.ix_(p_pos, h_pos)
                            ]
                            khat_sel = khat_profiles[q_row][np.ix_(p_pos, h_pos)]
                            ahat_block[q_row, h_pos] += np.einsum(
                                "p,ph,ph->h",
                                ff_profiles[q_row, p_pos],
                                projected_sel,
                                khat_sel,
                                optimize=True,
                            )

            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block
        return out

    def circular_fft_sparse_source_r_dependent(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = False,
        analytic_kernel_table_dx: float | None = None,
        q_block_size: int | None = None,
        profile_chunk_size: int | None = 64,
    ) -> np.ndarray:
        """Evaluate full cake map using sparse source projection and R cutoffs."""

        ahat = self.circular_ahat_sparse_source_r_dependent(
            q_indices=q_indices,
            form_factors=form_factors,
            margin=margin,
            cutoff_bin_size=cutoff_bin_size,
            analytic_kernel=analytic_kernel,
            analytic_kernel_table_dx=analytic_kernel_table_dx,
            q_block_size=q_block_size,
            profile_chunk_size=profile_chunk_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat_adaptive_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        dense_row_factor: float = 1.0,
        dense_batch_size: int = 2048,
    ) -> np.ndarray:
        """Return Fourier coefficients using row-adaptive sparse/dense beta paths."""

        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        (
            profile_e,
            profile_r,
            profile_z,
            _profile_starts,
            _profile_counts,
            _active_beta,
            _active_values,
        ) = self._sparse_profile_data()
        out = np.zeros((indices.size, self.binned.n_phi), dtype=self.complex_dtype)
        if profile_e.size == 0:
            self._last_adaptive_profile_stats = {
                "blocks": 0,
                "n_profiles": 0,
            }
            return out

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        sparse_name = (
            "circular_contract_sparse_rz64"
            if self.complex_dtype == np.dtype("complex64")
            else "circular_contract_sparse_rz"
        )
        block_stats: list[dict[str, float | int]] = []

        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is None:
                h_indices = np.arange(self.binned.n_phi, dtype=np.intp)
            else:
                h_indices = hsel
                khat = np.take(khat, hsel, axis=-1)
            khat = np.ascontiguousarray(khat, dtype=self.complex_dtype)
            active_hhat = self._adaptive_profile_hhat(
                h_indices,
                dense_row_factor=dense_row_factor,
                dense_batch_size=dense_batch_size,
            )
            stats_key = (tuple(int(h) for h in h_indices), float(dense_row_factor))
            stats = dict(self._adaptive_profile_hhat_stats[stats_key])
            stats["q_count"] = int(sel.size)
            block_stats.append(stats)

            if cpp_solvers is not None:
                ahat_block = getattr(cpp_solvers, sparse_name)(
                    np.ascontiguousarray(profile_e, dtype=np.int64),
                    np.ascontiguousarray(profile_r, dtype=np.int64),
                    np.ascontiguousarray(profile_z, dtype=np.int64),
                    np.ascontiguousarray(active_hhat, dtype=self.complex_dtype),
                    np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                    khat,
                    np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
                )
            else:
                ahat_block = np.zeros((sel.size, h_indices.size), dtype=self.complex_dtype)
                for chunk_start in range(0, profile_e.size, 256):
                    chunk = slice(chunk_start, min(chunk_start + 256, profile_e.size))
                    e_chunk = profile_e[chunk]
                    r_chunk = profile_r[chunk]
                    z_chunk = profile_z[chunk]
                    h_chunk = active_hhat[chunk]
                    coeff = (
                        self.z_phase[sel[:, None], z_chunk[None, :]]
                        * ff[e_chunk[None, :], sel[:, None]]
                    )
                    khat_chunk = np.take(khat, r_chunk, axis=1)
                    ahat_block += np.einsum(
                        "bc,bch,ch->bh",
                        coeff,
                        khat_chunk,
                        h_chunk,
                        optimize=True,
                    )

            if hsel is None:
                out[local] = ahat_block
            else:
                out[local] = 0.0
                out[local, hsel] = ahat_block

        h_counts = np.array([float(s["n_h"]) for s in block_stats], dtype=float)
        dense_fracs = np.array(
            [float(s["dense_profile_fraction"]) for s in block_stats],
            dtype=float,
        )
        dense_counts = np.array(
            [float(s["dense_profile_count"]) for s in block_stats],
            dtype=float,
        )
        self._last_adaptive_profile_stats = {
            "blocks": int(len(block_stats)),
            "n_profiles": int(block_stats[0]["n_profiles"]),
            "min_h": int(np.min(h_counts)),
            "max_h": int(np.max(h_counts)),
            "mean_h": float(np.mean(h_counts)),
            "min_dense_profile_fraction": float(np.min(dense_fracs)),
            "max_dense_profile_fraction": float(np.max(dense_fracs)),
            "mean_dense_profile_fraction": float(np.mean(dense_fracs)),
            "min_dense_profile_count": int(np.min(dense_counts)),
            "max_dense_profile_count": int(np.max(dense_counts)),
            "mean_dense_profile_count": float(np.mean(dense_counts)),
            "dense_row_factor": float(dense_row_factor),
        }
        return out

    def circular_fft_adaptive_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        dense_row_factor: float = 1.0,
        dense_batch_size: int = 2048,
    ) -> np.ndarray:
        """Evaluate circular FFT with row-adaptive sparse/dense beta transforms."""

        self._check_circular_grid()
        ahat = self.circular_ahat_adaptive_profiles(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
            dense_row_factor=dense_row_factor,
            dense_batch_size=dense_batch_size,
        )
        return np.fft.ifft(ahat, axis=-1)

    def circular_fft_with_form_factors(
        self,
        form_factors: FormFactors,
        q_indices: np.ndarray | None = None,
        *,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate circular FFT with replacement form factors on the same grid."""

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        ff = normalize_form_factors(self.binned.elements, self.q, form_factors)
        ahat = self._circular_ahat(indices, form_factors=ff, q_block_size=q_block_size)
        return np.fft.ifft(ahat, axis=-1)

    def circular_ahat(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return Fourier-domain ring coefficients without final IFFT."""

        indices = _normalize_q_indices(q_indices, self.q.size)
        ff = (
            None
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors)
        )
        return self._circular_ahat(indices, form_factors=ff, q_block_size=q_block_size)

    def ring_average_intensity(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return ``mean_phi |A(q, phi)|^2`` directly from Fourier coefficients."""

        return self.ring_average_intensity_r_grouped(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
        )

    def ring_average_intensity_fused(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return 1D intensity with a fused C++ coefficient contraction when available."""

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        if cpp_solvers is None:
            ahat = self._circular_ahat(indices, form_factors=ff, q_block_size=block_size)
            return np.sum(np.abs(ahat) ** 2, axis=-1) / (self.binned.n_phi**2)

        func_name = (
            "circular_ring_average_fused64"
            if self.complex_dtype == np.dtype("complex64")
            else "circular_ring_average_fused"
        )
        func = getattr(cpp_solvers, func_name)
        out = np.empty(indices.size, dtype=float)
        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is None:
                hhat = self.hhat
                normalization_scale = 1.0
            else:
                hhat = np.take(self.hhat, hsel, axis=-1)
                khat = np.take(khat, hsel, axis=-1)
                normalization_scale = (hhat.shape[-1] / self.binned.n_phi) ** 2
            block = func(
                np.ascontiguousarray(hhat, dtype=self.complex_dtype),
                np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                np.ascontiguousarray(khat, dtype=self.complex_dtype),
                np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
            )
            out[local] = block * normalization_scale
        return out

    def ring_average_intensity_via_ahat(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return 1D intensity through the generic ``Ahat(q,h)`` path."""

        ahat = self.circular_ahat(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
        )
        return np.sum(np.abs(ahat) ** 2, axis=-1) / (self.binned.n_phi**2)

    def ring_average_intensity_r_grouped(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return 1D intensity using explicit ``B(q, R, h)`` reduction.

        This follows the canonical order
        ``Hhat(e,R,z,h) -> B(q,R,h) -> Ahat(q,h) -> I(q)`` and avoids creating
        the final cake-map amplitude.
        """

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        out = np.empty(indices.size, dtype=float)
        for start in range(0, indices.size, block_size):
            local = slice(start, min(start + block_size, indices.size))
            sel = indices[local]
            khat = self._kernel_hat_block(sel)
            hsel = self._harmonic_indices_for_block(sel)
            if hsel is None:
                hhat = self.hhat
            else:
                hhat = np.take(self.hhat, hsel, axis=-1)
                khat = np.take(khat, hsel, axis=-1)

            if hhat.shape[0] == 1:
                b_rh = np.einsum(
                    "bz,rzh->brh",
                    self.z_phase[sel],
                    hhat[0],
                    optimize=True,
                ).astype(self.complex_dtype, copy=False)
                b_rh *= ff[0, sel, None, None]
            else:
                z_reduced = np.einsum(
                    "bz,erzh->berh",
                    self.z_phase[sel],
                    hhat,
                    optimize=True,
                )
                b_rh = np.einsum(
                    "eb,berh->brh",
                    ff[:, sel],
                    z_reduced,
                    optimize=True,
                ).astype(self.complex_dtype, copy=False)
            ahat = np.einsum("brh,brh->bh", b_rh, khat, optimize=True)
            out[local] = np.sum(np.abs(ahat) ** 2, axis=-1).real / (
                self.binned.n_phi**2
            )
        return out

    def ring_average_intensity_r_dependent_bandlimit(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return 1D intensity with an R-dependent harmonic cutoff.

        For each ``q`` and radial shell, only modes with
        ``|h| <= q_perp(q) * R + margin`` are accumulated. Cutoffs are rounded
        up to ``cutoff_bin_size`` so nearby R rows share vectorized reductions.
        This is an approximate fast path; compare against
        ``ring_average_intensity_r_grouped`` for error.
        """

        self._check_circular_grid()
        indices = _normalize_q_indices(q_indices, self.q.size)
        margin = int(margin)
        if margin < 0:
            raise ValueError("margin must be non-negative")
        cutoff_bin_size = int(cutoff_bin_size)
        if cutoff_bin_size <= 0:
            raise ValueError("cutoff_bin_size must be positive")

        ff = (
            self.form_factors
            if form_factors is None
            else normalize_form_factors(self.binned.elements, self.q, form_factors).astype(
                self.complex_dtype,
                copy=False,
            )
        )
        if ff.shape != self.form_factors.shape:
            raise ValueError("form_factors must have shape (n_elements, n_q)")

        n_phi = self.binned.n_phi
        block_size = self.q_block_size if q_block_size is None else int(q_block_size)
        if block_size <= 0:
            raise ValueError("q_block_size must be positive")

        use_cpp = self.circular_backend in {"auto", "cpp"}
        cpp_solvers = (
            _cpp_solver_module(required=self.circular_backend == "cpp")
            if use_cpp
            else None
        )
        if cpp_solvers is not None:
            func_name = (
                "circular_ring_average_r_dependent64"
                if self.complex_dtype == np.dtype("complex64")
                else "circular_ring_average_r_dependent"
            )
            func = getattr(cpp_solvers, func_name)
            out = np.empty(indices.size, dtype=float)
            hhat = np.ascontiguousarray(self.hhat, dtype=self.complex_dtype)
            for start in range(0, indices.size, block_size):
                local = slice(start, min(start + block_size, indices.size))
                sel = indices[local]
                out[local] = func(
                    hhat,
                    np.ascontiguousarray(self.z_phase[sel], dtype=self.complex_dtype),
                    np.ascontiguousarray(
                        self._kernel_hat_block(sel),
                        dtype=self.complex_dtype,
                    ),
                    np.ascontiguousarray(ff[:, sel], dtype=self.complex_dtype),
                    self._r_dependent_cutoff_matrix(
                        sel,
                        margin=margin,
                        cutoff_bin_size=cutoff_bin_size,
                    ),
                )
            return out

        modes = np.fft.fftfreq(n_phi, d=1.0 / n_phi).astype(int)
        abs_modes = np.abs(modes)
        out = np.empty(indices.size, dtype=float)

        for out_i, iq in enumerate(indices):
            khat = self._kernel_hat_block(np.array([iq], dtype=np.intp))[0]
            cutoffs = self._r_dependent_cutoff_matrix(
                np.array([iq], dtype=np.intp),
                margin=margin,
                cutoff_bin_size=cutoff_bin_size,
            )[0]

            ahat = np.zeros(n_phi, dtype=self.complex_dtype)
            for cutoff in np.unique(cutoffs):
                r_idx = np.flatnonzero(cutoffs == cutoff)
                if r_idx.size == 0:
                    continue
                if cutoff >= self.n_half:
                    hsel = np.arange(n_phi, dtype=np.intp)
                else:
                    hsel = np.flatnonzero(abs_modes <= cutoff).astype(
                        np.intp,
                        copy=False,
                    )
                if hsel.size == 0:
                    continue

                if self.hhat.shape[0] == 1:
                    hhat_r = np.take(
                        np.take(self.hhat[0], r_idx, axis=0),
                        hsel,
                        axis=-1,
                    )
                    b_rh = np.einsum(
                        "z,rzh->rh",
                        self.z_phase[iq],
                        hhat_r,
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)
                    b_rh *= ff[0, iq]
                else:
                    hhat_r = np.take(
                        np.take(self.hhat, r_idx, axis=1),
                        hsel,
                        axis=-1,
                    )
                    z_reduced = np.einsum(
                        "z,erzh->erh",
                        self.z_phase[iq],
                        hhat_r,
                        optimize=True,
                    )
                    b_rh = np.einsum(
                        "e,erh->rh",
                        ff[:, iq],
                        z_reduced,
                        optimize=True,
                    ).astype(self.complex_dtype, copy=False)

                khat_r = np.take(np.take(khat, r_idx, axis=0), hsel, axis=-1)
                ahat[hsel] += np.einsum("rh,rh->h", b_rh, khat_r, optimize=True)

            out[out_i] = np.sum(np.abs(ahat) ** 2).real / (n_phi**2)
        return out

    def ring_average_intensity_sparse_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
    ) -> np.ndarray:
        """Return ``mean_phi |A(q, phi)|^2`` from sparse profile coefficients."""

        ahat = self.circular_ahat_sparse_profiles(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
        )
        return np.sum(np.abs(ahat) ** 2, axis=-1) / (self.binned.n_phi**2)

    def ring_average_intensity_adaptive_profiles(
        self,
        q_indices: np.ndarray | None = None,
        *,
        form_factors: FormFactors = None,
        q_block_size: int | None = None,
        dense_row_factor: float = 1.0,
        dense_batch_size: int = 2048,
    ) -> np.ndarray:
        """Return ring-averaged intensity with row-adaptive profile transforms."""

        ahat = self.circular_ahat_adaptive_profiles(
            q_indices=q_indices,
            form_factors=form_factors,
            q_block_size=q_block_size,
            dense_row_factor=dense_row_factor,
            dense_batch_size=dense_batch_size,
        )
        return np.sum(np.abs(ahat) ** 2, axis=-1) / (self.binned.n_phi**2)

    def _ensure_harmonics(self, max_cutoff: int) -> None:
        max_cutoff = int(max_cutoff)
        if max_cutoff <= self._harmonic_max:
            return
        self._harmonic_max = max_cutoff
        self._modes = np.arange(-max_cutoff, max_cutoff + 1)
        indices = np.mod(self._modes, self.binned.n_phi)
        coeff = np.take(self.hhat, indices, axis=-1)
        self._hcoef = coeff * np.exp(-0.5j * self.delta_phi * self._modes)
        self._phi_basis = np.exp(1j * self._modes[:, None] * self.phi[None, :])
        self._i_to_m = np.exp(0.5j * np.pi * self._modes)

    def _normalize_cutoffs(
        self, harmonic_cutoff: int | np.ndarray | None, indices: np.ndarray
    ) -> np.ndarray:
        if harmonic_cutoff is None:
            return self.cutoffs[indices]

        cutoffs = np.asarray(harmonic_cutoff, dtype=int)
        if cutoffs.ndim == 0:
            out = np.full(indices.size, int(cutoffs), dtype=int)
        elif cutoffs.shape == (self.q.size,):
            out = cutoffs[indices]
        elif cutoffs.shape == (indices.size,):
            out = cutoffs
        else:
            raise ValueError(
                "harmonic_cutoff must be a scalar, one value per q, "
                "or one value per selected q"
            )
        return np.clip(out, 0, self.n_half)

    def jacobi_anger(
        self,
        q_indices: np.ndarray | None = None,
        *,
        harmonic_cutoff: int | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the Jacobi-Anger harmonic solver using cached coefficients."""

        indices = _normalize_q_indices(q_indices, self.q.size)
        cutoffs = self._normalize_cutoffs(harmonic_cutoff, indices)
        max_cutoff = int(cutoffs.max(initial=0))
        self._ensure_harmonics(max_cutoff)

        out = np.empty((indices.size, self.phi.size), dtype=np.complex128)
        abs_modes = np.abs(self.modes)
        for out_i, iq in enumerate(indices):
            keep = abs_modes <= cutoffs[out_i]
            modes_i = self.modes[keep]
            bessel = special.jv(
                modes_i[None, :],
                self.q_perp[iq] * self.binned.r_centers[:, None],
            )
            coeff = np.einsum(
                "e,z,erzm,rm,m->m",
                self.form_factors[:, iq],
                self.z_phase[iq],
                self.hcoef[..., keep],
                bessel,
                self.i_to_m[keep],
                optimize=True,
            )
            out[out_i] = coeff @ self.phi_basis[keep]
        return out, cutoffs

    def hybrid(
        self,
        q_indices: np.ndarray | None = None,
        *,
        switch_fraction: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use Jacobi-Anger at low harmonic coverage and circular FFT otherwise."""

        indices = _normalize_q_indices(q_indices, self.q.size)
        cutoffs = self.cutoffs[indices]
        fraction = self.switch_fraction if switch_fraction is None else float(
            switch_fraction
        )
        use_jacobi = cutoffs <= int(np.floor(fraction * self.n_half))

        if np.all(use_jacobi):
            out, _ = self.jacobi_anger(q_indices=indices, harmonic_cutoff=cutoffs)
            return out, cutoffs, use_jacobi
        if not np.any(use_jacobi):
            out = self.circular_fft(q_indices=indices)
            return out, cutoffs, use_jacobi

        out = np.empty((indices.size, self.phi.size), dtype=np.complex128)
        if np.any(use_jacobi):
            jac, _ = self.jacobi_anger(
                q_indices=indices[use_jacobi],
                harmonic_cutoff=cutoffs[use_jacobi],
            )
            out[use_jacobi] = jac
        if np.any(~use_jacobi):
            out[~use_jacobi] = self.circular_fft(q_indices=indices[~use_jacobi])
        return out, cutoffs, use_jacobi


class KernelInterpolationTable:
    """Linear interpolation table for ``FFT(exp(i x cos(phi)))``."""

    def __init__(
        self,
        n_phi: int,
        x_max: float,
        dx: float,
        *,
        complex_dtype: np.dtype | str = np.complex128,
    ) -> None:
        self.n_phi = int(n_phi)
        self.dx = float(dx)
        self.complex_dtype = np.dtype(complex_dtype)
        if self.complex_dtype not in (np.dtype("complex64"), np.dtype("complex128")):
            raise ValueError("complex_dtype must be complex64 or complex128")
        if self.n_phi <= 0:
            raise ValueError("n_phi must be positive")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        x_max = max(0.0, float(x_max))
        self.x_grid = np.arange(0.0, x_max + 2.0 * self.dx, self.dx)
        angles = np.arange(self.n_phi) * (2.0 * np.pi / self.n_phi)
        phase = self.x_grid[:, None] * np.cos(angles)[None, :]
        kernel = _unit_complex_from_phase_dtype(phase, self.complex_dtype)
        self.table = scipy_fft.fft(kernel, axis=-1, workers=1)

    def khat(self, q_perp: np.ndarray, r_centers: np.ndarray) -> np.ndarray:
        x = np.asarray(q_perp)[:, None] * np.asarray(r_centers)[None, :]
        scaled = x / self.dx
        lower = np.floor(scaled).astype(np.int64)
        lower = np.clip(lower, 0, self.table.shape[0] - 2)
        frac = scaled - lower
        out = (
            self.table[lower] * (1.0 - frac[..., None])
            + self.table[lower + 1] * frac[..., None]
        )
        return out.astype(self.complex_dtype, copy=False)


def jacobi_anger_amplitude(
    binned: BinnedStructure,
    q: np.ndarray,
    wavelength: float,
    *,
    phi: np.ndarray | None = None,
    form_factors: FormFactors = None,
    harmonic_cutoff: int | np.ndarray | None = None,
    cutoff_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Binned Jacobi-Anger expansion on Ewald rings.

    Returns ``(amplitude, cutoffs)``. When ``harmonic_cutoff`` is ``None``, a
    q-dependent Bessel cutoff is estimated from ``q_perp * r_max`` and capped by
    the angular Nyquist limit ``n_phi // 2``.
    """

    plan = PreparedCakePlan(
        binned,
        q,
        wavelength,
        phi=phi,
        form_factors=form_factors,
        cutoff_tol=cutoff_tol,
    )
    return plan.jacobi_anger(harmonic_cutoff=harmonic_cutoff)


def hybrid_amplitude(
    binned: BinnedStructure,
    q: np.ndarray,
    wavelength: float,
    *,
    phi: np.ndarray | None = None,
    form_factors: FormFactors = None,
    cutoff_tol: float = 1e-8,
    switch_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use Jacobi-Anger at low harmonic coverage and circular FFT otherwise."""

    plan = PreparedCakePlan(
        binned,
        q,
        wavelength,
        phi=phi,
        form_factors=form_factors,
        cutoff_tol=cutoff_tol,
        switch_fraction=switch_fraction,
    )
    return plan.hybrid()


def nufft_amplitude(
    coords: np.ndarray,
    q: np.ndarray,
    wavelength: float,
    phi: np.ndarray,
    *,
    atom_weights: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """FINUFFT type-3 reference for q-independent atom weights."""

    try:
        import finufft
    except ImportError as exc:
        raise RuntimeError("finufft is not installed") from exc

    coords = np.asarray(coords, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if atom_weights is None:
        weights = np.ones(coords.shape[0], dtype=np.complex128)
    else:
        weights = np.asarray(atom_weights, dtype=np.complex128)

    q_perp, q_z = ewald_ring(q, wavelength)
    qx = (q_perp[:, None] * np.cos(phi)[None, :]).ravel()
    qy = (q_perp[:, None] * np.sin(phi)[None, :]).ravel()
    qz = np.broadcast_to(q_z[:, None], (q.size, phi.size)).ravel()
    values = finufft.nufft3d3(
        np.ascontiguousarray(coords[:, 0]),
        np.ascontiguousarray(coords[:, 1]),
        np.ascontiguousarray(coords[:, 2]),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(qx),
        np.ascontiguousarray(qy),
        np.ascontiguousarray(qz),
        eps=eps,
        isign=1,
    )
    return values.reshape(q.size, phi.size)


def nufft_amplitude_chunked(
    coords: np.ndarray,
    q: np.ndarray,
    wavelength: float,
    phi: np.ndarray,
    *,
    atom_weights: np.ndarray | None = None,
    eps: float = 1e-9,
    q_block_size: int = 1,
) -> np.ndarray:
    """FINUFFT type-3 reference evaluated in q-blocks to limit plan memory."""

    q = np.asarray(q, dtype=float)
    block_size = int(q_block_size)
    if block_size <= 0:
        raise ValueError("q_block_size must be positive")
    out = np.empty((q.size, np.asarray(phi).size), dtype=np.complex128)
    for start in range(0, q.size, block_size):
        stop = min(start + block_size, q.size)
        out[start:stop] = nufft_amplitude(
            coords,
            q[start:stop],
            wavelength,
            phi,
            atom_weights=atom_weights,
            eps=eps,
        )
    return out
