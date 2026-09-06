"""Prepared SO(2) dyadic Green blocks for an isotropic planar interface.

The reflected Maxwell Green tensor is represented as an azimuthal Fourier
series of TE/TM polarization dyads.  Arbitrary source--target offsets are
allowed; the interface normal supplies the residual SO(2) symmetry.  The
prepared cache contains the base tensor and its Cartesian phase generators,
so field application, source adjoints, and geometry JVP/VJP operations use the
same quadrature and harmonic support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import special

from .harmonic_support import minimum_symmetric_cutoff, signed_angular_modes
from .solvers import estimate_bessel_cutoff
from .vector_debye import cylindrical_bessel_band_miller


Array = np.ndarray
BesselBackend = Literal["miller", "scipy"]


@dataclass(frozen=True)
class LayeredDyadicHarmonicContract:
    """Frozen support contract for a layered dyadic Green cache."""

    n_phi: int
    kernel_cutoff: int
    tensor_cutoff: int
    mode_padding: int
    required_cutoff: int
    compute_cutoff: int
    symmetric_nyquist: int
    nyquist_limited: bool
    kernel_tolerance: float
    tensor_tail_tolerance: float
    miller_margin: int
    bessel_backend: str
    voxel_side: float
    voxel_rotation: float

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def isotropic_reflection_dyad(
    radial_wavenumbers: Any,
    phi: Any,
    *,
    upper_wavenumber: float,
    lower_wavenumber: float,
    damping: float,
) -> tuple[Array, Array, Array, Array]:
    r"""Return the reflected TE/TM dyad on a cylindrical spectral grid.

    The returned tensor is

    ``r_s s s^T + r_p p_plus p_minus^T``

    for an upward field produced by a downward source spectrum in the upper
    half-space.  The electric-field Fresnel coefficient is used for ``r_p``.
    """

    radial = _finite_vector("radial_wavenumbers", radial_wavenumbers)
    phi = _finite_vector("phi", phi)
    if np.any(radial < 0.0):
        raise ValueError("radial_wavenumbers must be non-negative")
    upper_wavenumber = float(upper_wavenumber)
    lower_wavenumber = float(lower_wavenumber)
    damping = float(damping)
    if (
        not np.isfinite(upper_wavenumber)
        or not np.isfinite(lower_wavenumber)
        or not np.isfinite(damping)
        or upper_wavenumber <= 0.0
        or lower_wavenumber <= 0.0
        or damping <= 0.0
    ):
        raise ValueError("wavenumbers and damping must be finite and positive")

    k1 = complex(upper_wavenumber, damping)
    k2 = complex(lower_wavenumber, damping)
    kz1 = _outgoing_root(k1**2 - radial.astype(np.complex128) ** 2)
    kz2 = _outgoing_root(k2**2 - radial.astype(np.complex128) ** 2)
    rs = (kz1 - kz2) / (kz1 + kz2)
    epsilon1 = k1**2
    epsilon2 = k2**2
    rp = (epsilon2 * kz1 - epsilon1 * kz2) / (
        epsilon2 * kz1 + epsilon1 * kz2
    )

    cos_phi = np.cos(phi)[None, :]
    sin_phi = np.sin(phi)[None, :]
    ones = np.ones((radial.size, phi.size), dtype=np.complex128)
    zeros = np.zeros_like(ones)
    s = np.stack(
        np.broadcast_arrays(-sin_phi, cos_phi, zeros), axis=-1
    ).astype(np.complex128, copy=False)
    sin_theta = radial[:, None] / k1
    cos_theta = kz1[:, None] / k1
    p_plus = np.stack(
        (
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta * np.ones_like(cos_phi),
        ),
        axis=-1,
    )
    p_minus = np.stack(
        (
            -cos_theta * cos_phi,
            -cos_theta * sin_phi,
            -sin_theta * np.ones_like(cos_phi),
        ),
        axis=-1,
    )
    s_outer = s[..., :, None] * s[..., None, :]
    p_outer = p_plus[..., :, None] * p_minus[..., None, :]
    dyad = rs[:, None, None, None] * s_outer + rp[:, None, None, None] * p_outer
    return dyad, kz1, rs, rp


def uniform_voxel_pair_form_factor(
    radial_wavenumbers: Any,
    phi: Any,
    axial_wavenumbers: Any,
    *,
    voxel_side: float,
    voxel_rotation: float = 0.0,
) -> Array:
    r"""Return the source--target form factor for equal uniform cubes.

    The reflected angular-spectrum phase contains
    ``exp(i k_parallel . (r_t-r_s) + i k_z (z_t+z_s))``.  Averaging this
    phase over a uniformly polarized source cube and a uniformly tested
    target cube therefore contributes two lateral sinc factors and two axial
    sinc factors.  The axial pair is a square, not an absolute square, because
    source and target heights enter the reflected phase with the same sign.

    ``voxel_side=0`` recovers point collocation exactly.
    """

    radial = _finite_vector("radial_wavenumbers", radial_wavenumbers)
    phi_values = _finite_vector("phi", phi)
    axial = np.asarray(axial_wavenumbers, dtype=np.complex128)
    if axial.shape != radial.shape or not np.all(np.isfinite(axial)):
        raise ValueError(
            "axial_wavenumbers must be a finite vector matching radial_wavenumbers"
        )
    side = float(voxel_side)
    rotation = float(voxel_rotation)
    if not np.isfinite(side) or side < 0.0:
        raise ValueError("voxel_side must be finite and non-negative")
    if not np.isfinite(rotation):
        raise ValueError("voxel_rotation must be finite")
    if side == 0.0:
        return np.ones((radial.size, phi_values.size), dtype=np.complex128)

    relative_phi = phi_values[None, :] - rotation
    half_side = 0.5 * side
    kx = radial[:, None] * np.cos(relative_phi)
    ky = radial[:, None] * np.sin(relative_phi)
    lateral = _complex_sinc(half_side * kx) ** 2 * _complex_sinc(
        half_side * ky
    ) ** 2
    axial_pair = _complex_sinc(half_side * axial)[:, None] ** 2
    return np.asarray(lateral * axial_pair, dtype=np.complex128)


@dataclass(frozen=True)
class PreparedLayeredDyadicGreenOperator:
    """Prepared reflected dyadic Green map for arbitrary interface offsets.

    ``x``, ``y`` and ``height`` describe target-minus-source lateral offsets
    and the positive source-plus-target interface height.  ``apply`` maps one
    complex electric dipole vector to a three-component field at all prepared
    offsets.  The geometry derivatives are with respect to these three stored
    coordinates.
    """

    radial_wavenumbers: Array
    quadrature_weights: Array
    phi: Array
    x: Array
    y: Array
    height: Array
    upper_wavenumber: float
    lower_wavenumber: float
    damping: float
    voxel_side: float
    voxel_rotation: float
    axial_upper: Array
    reflection_s: Array
    reflection_p: Array
    spectral_generators: Array
    h: Array
    mode_mask: Array
    radial_basis: Array
    angular_basis: Array
    axial_weights: Array
    contract: LayeredDyadicHarmonicContract

    @classmethod
    def build(
        cls,
        *,
        radial_wavenumbers: Any,
        quadrature_weights: Any,
        phi: Any,
        x: Any,
        y: Any,
        height: Any,
        upper_wavenumber: float,
        lower_wavenumber: float,
        damping: float,
        voxel_side: float = 0.0,
        voxel_rotation: float = 0.0,
        h_cutoff: int | None = None,
        kernel_tolerance: float = 1.0e-12,
        tensor_tail_tolerance: float = 1.0e-13,
        mode_padding: int = 8,
        miller_margin: int = 32,
        bessel_backend: BesselBackend = "miller",
        allow_underresolved: bool = False,
    ) -> "PreparedLayeredDyadicGreenOperator":
        radial = _finite_vector("radial_wavenumbers", radial_wavenumbers)
        weights = _finite_vector("quadrature_weights", quadrature_weights)
        phi = _uniform_periodic_phi(phi)
        if weights.shape != radial.shape or np.any(radial < 0.0) or np.any(weights <= 0.0):
            raise ValueError(
                "radial_wavenumbers and positive quadrature_weights must have equal shapes"
            )
        x_array, y_array, height_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(height, dtype=np.float64),
        )
        if x_array.size == 0 or not (
            np.all(np.isfinite(x_array))
            and np.all(np.isfinite(y_array))
            and np.all(np.isfinite(height_array))
        ):
            raise ValueError("x, y and height must broadcast to non-empty finite arrays")
        if np.any(height_array <= 0.0):
            raise ValueError("height must be positive")
        if not 0.0 < float(kernel_tolerance) < 1.0:
            raise ValueError("kernel_tolerance must lie in (0, 1)")
        if not 0.0 < float(tensor_tail_tolerance) < 1.0:
            raise ValueError("tensor_tail_tolerance must lie in (0, 1)")
        mode_padding = int(mode_padding)
        miller_margin = int(miller_margin)
        if mode_padding < 0 or miller_margin < 0:
            raise ValueError("mode_padding and miller_margin must be non-negative")
        if bessel_backend not in {"miller", "scipy"}:
            raise ValueError("bessel_backend must be 'miller' or 'scipy'")
        voxel_side = float(voxel_side)
        voxel_rotation = float(voxel_rotation)
        if not np.isfinite(voxel_side) or voxel_side < 0.0:
            raise ValueError("voxel_side must be finite and non-negative")
        if not np.isfinite(voxel_rotation):
            raise ValueError("voxel_rotation must be finite")

        rho = np.hypot(x_array, y_array).ravel()
        psi = np.arctan2(y_array, x_array).ravel()
        self_only = bool(np.all(rho == 0.0))

        dyad, kz1, rs, rp = isotropic_reflection_dyad(
            radial,
            phi,
            upper_wavenumber=upper_wavenumber,
            lower_wavenumber=lower_wavenumber,
            damping=damping,
        )
        voxel_form_factor = uniform_voxel_pair_form_factor(
            radial,
            phi,
            kz1,
            voxel_side=voxel_side,
            voxel_rotation=voxel_rotation,
        )
        dyad = dyad * voxel_form_factor[:, :, None, None]
        q = radial[:, None, None, None]
        cos_phi = np.cos(phi)[None, :, None, None]
        sin_phi = np.sin(phi)[None, :, None, None]
        if self_only:
            # Only h=0 survives because J_h(0)=delta_h0.  Forming the full
            # angular FFT would spend memory on coefficients that the exact
            # contraction annihilates.
            spectral_mean = np.stack(
                (
                    np.mean(dyad, axis=1),
                    np.mean(1j * q * cos_phi * dyad, axis=1),
                    np.mean(1j * q * sin_phi * dyad, axis=1),
                    np.mean(1j * kz1[:, None, None, None] * dyad, axis=1),
                ),
                axis=0,
            )[:, :, None, :, :]
            generator_fourier = None
            tensor_cutoff = 0
        else:
            generators = np.stack(
                (
                    dyad,
                    1j * q * cos_phi * dyad,
                    1j * q * sin_phi * dyad,
                    1j * kz1[:, None, None, None] * dyad,
                ),
                axis=0,
            )
            generator_fourier = np.fft.fft(generators, axis=2)
            tensor_cutoff = minimum_symmetric_cutoff(
                generator_fourier,
                tensor_tail_tolerance,
                axis=2,
            )
        x_max = float(np.max(radial) * np.max(rho))
        kernel_cutoff = estimate_bessel_cutoff(x_max, tol=kernel_tolerance)
        # The azimuthal integral is performed after Fourier expanding the
        # TE/TM tensor, not after sampling the translation phase.  Therefore
        # only tensor-generator modes are retained.  Large q*rho changes the
        # Bessel arguments and hence Miller's downward-recursion start, but it
        # does not create new tensor Fourier coefficients.  ``kernel_cutoff``
        # remains a diagnostic for a sampled-phase representation.
        # At zero lateral offset J_h(0)=0 for every h != 0.  A batch made
        # entirely of reflected self offsets therefore contracts exactly to
        # the azimuthal mean even when the uniform-voxel spectrum itself has
        # high Fourier content.
        required_cutoff = 0 if self_only else tensor_cutoff + mode_padding
        symmetric_nyquist = (phi.size - 1) // 2
        raw_requested_cutoff = required_cutoff if h_cutoff is None else int(h_cutoff)
        if raw_requested_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")
        requested_cutoff = 0 if self_only else raw_requested_cutoff
        underresolved = requested_cutoff < required_cutoff
        nyquist_limited = (
            required_cutoff > symmetric_nyquist
            or requested_cutoff > symmetric_nyquist
        )
        if (underresolved or nyquist_limited) and not allow_underresolved:
            if underresolved:
                reason = (
                    f"requested H={requested_cutoff} is below required H={required_cutoff}"
                )
            else:
                reason = (
                    f"requested/required H={max(requested_cutoff, required_cutoff)} "
                    f"exceeds symmetric Nyquist H={symmetric_nyquist}"
                )
            raise ValueError(reason + "; increase n_phi or explicitly allow under-resolution")
        compute_cutoff = min(requested_cutoff, symmetric_nyquist)
        modes = signed_angular_modes(phi.size)
        mode_mask = np.abs(modes) <= compute_cutoff
        h = modes[mode_mask]
        if self_only:
            spectral_generators = spectral_mean
        else:
            assert generator_fourier is not None
            spectral_generators = generator_fourier[:, :, mode_mask, :, :] / float(
                phi.size
            )

        arguments = radial[:, None] * rho[None, :]
        abs_h = np.abs(h)
        unique_abs_h, inverse = np.unique(abs_h, return_inverse=True)
        if bessel_backend == "miller":
            band = cylindrical_bessel_band_miller(
                arguments,
                int(np.max(unique_abs_h, initial=0)),
                margin=miller_margin,
            )
            bessel = band[:, :, unique_abs_h]
        else:
            bessel = special.jv(
                unique_abs_h[None, None, :], arguments[:, :, None]
            )
        radial_basis = np.transpose(bessel[:, :, inverse], (0, 2, 1))
        radial_basis = radial_basis * np.power(1j, abs_h)[None, :, None]
        angular_basis = np.exp(
            1j * h[:, None] * (psi[None, :] - float(phi[0]))
        )
        complex_weight = (
            1j
            / (4.0 * np.pi)
            * weights.astype(np.complex128)
            * radial.astype(np.complex128)
            / kz1
        )
        axial_phase = np.exp(
            1j * kz1[:, None] * height_array.ravel()[None, :]
        )
        axial_weights = complex_weight[:, None] * axial_phase
        contract = LayeredDyadicHarmonicContract(
            n_phi=int(phi.size),
            kernel_cutoff=int(kernel_cutoff),
            tensor_cutoff=int(tensor_cutoff),
            mode_padding=mode_padding,
            required_cutoff=int(required_cutoff),
            compute_cutoff=int(compute_cutoff),
            symmetric_nyquist=int(symmetric_nyquist),
            nyquist_limited=bool(nyquist_limited or underresolved),
            kernel_tolerance=float(kernel_tolerance),
            tensor_tail_tolerance=float(tensor_tail_tolerance),
            miller_margin=miller_margin,
            bessel_backend=bessel_backend,
            voxel_side=voxel_side,
            voxel_rotation=voxel_rotation,
        )
        return cls(
            radial_wavenumbers=_readonly(radial),
            quadrature_weights=_readonly(weights),
            phi=_readonly(phi),
            x=_readonly(x_array.ravel()),
            y=_readonly(y_array.ravel()),
            height=_readonly(height_array.ravel()),
            upper_wavenumber=float(upper_wavenumber),
            lower_wavenumber=float(lower_wavenumber),
            damping=float(damping),
            voxel_side=voxel_side,
            voxel_rotation=voxel_rotation,
            axial_upper=_readonly(kz1),
            reflection_s=_readonly(rs),
            reflection_p=_readonly(rp),
            spectral_generators=_readonly(spectral_generators),
            h=_readonly(h),
            mode_mask=_readonly(mode_mask),
            radial_basis=_readonly(radial_basis),
            angular_basis=_readonly(angular_basis),
            axial_weights=_readonly(axial_weights),
            contract=contract,
        )

    @property
    def target_count(self) -> int:
        return int(self.x.size)

    @property
    def field_shape(self) -> tuple[int, int]:
        return (3, self.target_count)

    @property
    def cache_bytes(self) -> int:
        return int(
            self.spectral_generators.nbytes
            + self.radial_basis.nbytes
            + self.angular_basis.nbytes
            + self.axial_weights.nbytes
        )

    def kernel(self) -> Array:
        """Return the prepared reflected ``(field, source, target)`` tensor."""

        return self._contract(self.spectral_generators[0])

    def derivative_kernels(self) -> Array:
        """Return Cartesian ``x,y,height`` derivative kernels."""

        return np.stack(
            [self._contract(self.spectral_generators[index]) for index in range(1, 4)],
            axis=0,
        )

    def apply(self, source_vector: Any) -> Array:
        source = self._validate_source(source_vector)
        projected = np.einsum(
            "qhij,j->qhi", self.spectral_generators[0], source, optimize=True
        )
        return np.einsum(
            "qhi,qhu,hu,qu->iu",
            projected,
            self.radial_basis,
            self.angular_basis,
            self.axial_weights,
            optimize=True,
        )

    def adjoint(self, field_cotangent: Any) -> Array:
        cotangent = self._validate_field(field_cotangent)
        return np.einsum(
            "iju,iu->j", np.conjugate(self.kernel()), cotangent, optimize=True
        )

    def geometry_jvp(self, source_vector: Any, direction_xyz: Any) -> Array:
        """JVP for per-target perturbations of ``x,y,height``."""

        source = self._validate_source(source_vector)
        direction = self._validate_direction(direction_xyz)
        derivatives = np.empty((3, 3, self.target_count), dtype=np.complex128)
        for axis in range(3):
            projected = np.einsum(
                "qhij,j->qhi",
                self.spectral_generators[axis + 1],
                source,
                optimize=True,
            )
            derivatives[axis] = np.einsum(
                "qhi,qhu,hu,qu->iu",
                projected,
                self.radial_basis,
                self.angular_basis,
                self.axial_weights,
                optimize=True,
            )
        return np.einsum("aiu,ua->iu", derivatives, direction, optimize=True)

    def geometry_vjp(self, source_vector: Any, field_cotangent: Any) -> Array:
        """Real VJP for each prepared target's ``x,y,height`` coordinates."""

        source = self._validate_source(source_vector)
        cotangent = self._validate_field(field_cotangent)
        gradients = np.empty((self.target_count, 3), dtype=np.float64)
        for axis in range(3):
            derivative_field = np.einsum(
                "iju,j->iu",
                self._contract(self.spectral_generators[axis + 1]),
                source,
                optimize=True,
            )
            gradients[:, axis] = np.real(
                np.sum(np.conjugate(cotangent) * derivative_field, axis=0)
            )
        return gradients

    def direct_apply(
        self,
        source_vector: Any,
        *,
        x: Any | None = None,
        y: Any | None = None,
        height: Any | None = None,
    ) -> Array:
        """Direct sampled ``q,phi`` quadrature for representation parity."""

        source = self._validate_source(source_vector)
        x_values = self.x if x is None else np.asarray(x, dtype=np.float64)
        y_values = self.y if y is None else np.asarray(y, dtype=np.float64)
        height_values = self.height if height is None else np.asarray(height, dtype=np.float64)
        x_values, y_values, height_values = np.broadcast_arrays(
            x_values, y_values, height_values
        )
        if x_values.size != self.target_count:
            raise ValueError("direct coordinates must contain target_count entries")
        if not (
            np.all(np.isfinite(x_values))
            and np.all(np.isfinite(y_values))
            and np.all(np.isfinite(height_values))
            and np.all(height_values > 0.0)
        ):
            raise ValueError("direct coordinates must be finite with positive height")
        dyad, _, _, _ = isotropic_reflection_dyad(
            self.radial_wavenumbers,
            self.phi,
            upper_wavenumber=self.upper_wavenumber,
            lower_wavenumber=self.lower_wavenumber,
            damping=self.damping,
        )
        voxel_form_factor = uniform_voxel_pair_form_factor(
            self.radial_wavenumbers,
            self.phi,
            self.axial_upper,
            voxel_side=self.voxel_side,
            voxel_rotation=self.voxel_rotation,
        )
        dyad = dyad * voxel_form_factor[:, :, None, None]
        projected = np.einsum("qpij,j->qpi", dyad, source, optimize=True)
        q = self.radial_wavenumbers[:, None, None]
        phi = self.phi[None, :, None]
        phase = np.exp(
            1j
            * (
                q
                * (
                    np.cos(phi) * x_values.ravel()[None, None, :]
                    + np.sin(phi) * y_values.ravel()[None, None, :]
                )
                + self.axial_upper[:, None, None]
                * height_values.ravel()[None, None, :]
            )
        )
        spectral_weight = (
            1j
            / (8.0 * np.pi**2)
            * self.quadrature_weights.astype(np.complex128)
            * self.radial_wavenumbers.astype(np.complex128)
            / self.axial_upper
        )
        dphi = 2.0 * np.pi / float(self.phi.size)
        return dphi * np.einsum(
            "q,qpi,qpu->iu", spectral_weight, projected, phase, optimize=True
        )

    def _contract(self, spectral_coefficients: Array) -> Array:
        return np.einsum(
            "qhij,qhu,hu,qu->iju",
            spectral_coefficients,
            self.radial_basis,
            self.angular_basis,
            self.axial_weights,
            optimize=True,
        )

    def _validate_source(self, source_vector: Any) -> Array:
        source = np.asarray(source_vector, dtype=np.complex128)
        if source.shape != (3,) or not np.all(np.isfinite(source)):
            raise ValueError("source_vector must be a finite complex length-three vector")
        return source

    def _validate_field(self, field: Any) -> Array:
        values = np.asarray(field, dtype=np.complex128)
        if values.shape != self.field_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"field_cotangent must be finite with shape {self.field_shape}")
        return values

    def _validate_direction(self, direction_xyz: Any) -> Array:
        direction = np.asarray(direction_xyz, dtype=np.float64)
        if direction.shape == (3,):
            direction = np.broadcast_to(direction, (self.target_count, 3))
        if direction.shape != (self.target_count, 3) or not np.all(
            np.isfinite(direction)
        ):
            raise ValueError(
                "direction_xyz must be finite with shape (3,) or (target_count, 3)"
            )
        return direction


def _outgoing_root(value: Array) -> Array:
    root = np.sqrt(np.asarray(value, dtype=np.complex128))
    flip = (root.imag < 0.0) | ((root.imag == 0.0) & (root.real < 0.0))
    return np.where(flip, -root, root)


def _complex_sinc(value: Any) -> Array:
    """Return ``sin(z)/z`` with a stable removable singularity."""

    z = np.asarray(value, dtype=np.complex128)
    small = np.abs(z) < 1.0e-4
    z2 = z * z
    series = 1.0 - z2 / 6.0 + z2 * z2 / 120.0 - z2 * z2 * z2 / 5040.0
    safe = np.where(small, 1.0 + 0.0j, z)
    return np.where(small, series, np.sin(z) / safe)


def _finite_vector(name: str, values: Any) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return np.array(array, dtype=np.float64, copy=True)


def _uniform_periodic_phi(phi: Any) -> Array:
    values = _finite_vector("phi", phi)
    if values.size < 3:
        raise ValueError("phi must contain at least three samples")
    step = 2.0 * np.pi / float(values.size)
    if not np.allclose(np.diff(values), step, rtol=1.0e-12, atol=1.0e-13):
        raise ValueError("phi must be one complete uniform 2pi orbit")
    return values


def _readonly(values: Array) -> Array:
    array = np.ascontiguousarray(values)
    array.setflags(write=False)
    return array
