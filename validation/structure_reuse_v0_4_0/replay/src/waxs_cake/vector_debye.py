"""Prepared vector Debye--Wolf operators with paired differentiation.

The operator keeps the azimuthal Fourier representation explicit.  A single
prepared cache is shared by the forward map, its Euclidean adjoint, Jones-space
JVP/VJP operations, and rigid target-translation derivatives.  The object or
pupil need not be axisymmetric; only the sampled propagation orbit is assumed
to be a complete, uniform :math:`SO(2)` orbit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import special

from .harmonic_support import minimum_symmetric_cutoff, signed_angular_modes
from .solvers import estimate_bessel_cutoff


Array = np.ndarray
BesselBackend = Literal["miller", "scipy"]


def gauss_sine_theta_grid(ntheta: int, theta_max: float) -> tuple[Array, Array]:
    """Gauss--Legendre nodes for the ``sin(theta) dtheta`` measure."""

    ntheta = int(ntheta)
    theta_max = float(theta_max)
    if ntheta <= 0:
        raise ValueError("ntheta must be positive")
    if not 0.0 < theta_max <= np.pi:
        raise ValueError("theta_max must lie in (0, pi]")
    cos_max = float(np.cos(theta_max))
    nodes, weights = np.polynomial.legendre.leggauss(ntheta)
    half_width = 0.5 * (1.0 - cos_max)
    midpoint = 0.5 * (1.0 + cos_max)
    cos_theta = midpoint + half_width * nodes
    return np.arccos(cos_theta), half_width * weights


def richards_wolf_jones_matrix(
    theta: Array,
    phi: Array,
    *,
    apodization: Literal["none", "sqrt-cos"] = "sqrt-cos",
) -> Array:
    """Return the aplanatic map from transverse Jones data to ``Ex,Ey,Ez``."""

    theta = _finite_vector("theta", theta)
    phi = _finite_vector("phi", phi)
    theta_2d = theta[:, None]
    phi_2d = phi[None, :]
    cos_theta = np.cos(theta_2d)
    sin_theta = np.sin(theta_2d)
    cos_phi = np.cos(phi_2d)
    sin_phi = np.sin(phi_2d)

    matrix = np.empty((3, 2, theta.size, phi.size), dtype=np.complex128)
    matrix[0, 0] = cos_theta * cos_phi**2 + sin_phi**2
    matrix[0, 1] = (cos_theta - 1.0) * cos_phi * sin_phi
    matrix[1, 0] = matrix[0, 1]
    matrix[1, 1] = cos_theta * sin_phi**2 + cos_phi**2
    matrix[2, 0] = -sin_theta * cos_phi
    matrix[2, 1] = -sin_theta * sin_phi
    if apodization == "none":
        return matrix
    if apodization == "sqrt-cos":
        if np.any(cos_theta < -64.0 * np.finfo(float).eps):
            raise ValueError("sqrt-cos apodization requires theta <= pi/2")
        return matrix * np.sqrt(np.maximum(cos_theta, 0.0))[None, None, :, :]
    raise ValueError("apodization must be 'none' or 'sqrt-cos'")


def mix_jones_pupil(pupil_jones: Array, mixing: Array) -> Array:
    """Apply a sampled 3-by-2 polarization block to a Jones pupil."""

    pupil = np.asarray(pupil_jones, dtype=np.complex128)
    matrix = np.asarray(mixing, dtype=np.complex128)
    if pupil.ndim != 3 or pupil.shape[0] != 2:
        raise ValueError("pupil_jones must have shape (2, ntheta, nphi)")
    if matrix.shape != (3, 2, pupil.shape[1], pupil.shape[2]):
        raise ValueError("mixing must have shape (3, 2, ntheta, nphi)")
    return np.einsum("cjtp,jtp->ctp", matrix, pupil, optimize=True)


def unmix_jones_adjoint(effective_gradient: Array, mixing: Array) -> Array:
    """Apply the pointwise Hermitian transpose of ``mix_jones_pupil``."""

    gradient = np.asarray(effective_gradient, dtype=np.complex128)
    matrix = np.asarray(mixing, dtype=np.complex128)
    if gradient.ndim != 3 or gradient.shape[0] != 3:
        raise ValueError("effective_gradient must have shape (3, ntheta, nphi)")
    if matrix.shape != (3, 2, gradient.shape[1], gradient.shape[2]):
        raise ValueError("mixing shape does not match effective_gradient")
    return np.einsum(
        "cjtp,ctp->jtp", np.conjugate(matrix), gradient, optimize=True
    )


@dataclass(frozen=True)
class VectorDebyeHarmonicContract:
    """Frozen accuracy and storage contract for a prepared vector operator."""

    n_phi: int
    kernel_cutoff: int
    data_cutoff: int
    mode_padding: int
    required_cutoff: int
    compute_cutoff: int
    symmetric_nyquist: int
    nyquist_limited: bool
    kernel_tolerance: float
    source_tail_tolerance: float
    miller_margin: int
    bessel_backend: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedVectorDebyeOperator:
    """Prepared vector Richards--Wolf map on a separable cylindrical grid.

    The output storage order is ``(component, rho, psi, z)`` and ``forward``
    returns the last three axes flattened.  ``adjoint`` is paired under
    ``numpy.vdot`` with Euclidean weights on the sampled Jones pupil and target
    arrays; the physical ``sin(theta)dtheta dphi`` quadrature is already part
    of the forward cache.
    """

    theta: Array
    theta_weights: Array
    phi: Array
    rho_axis: Array
    psi_axis: Array
    z_axis: Array
    k: float
    mixing: Array
    h: Array
    mode_mask: Array
    radial: Array
    angular: Array
    defocus: Array
    contract: VectorDebyeHarmonicContract

    @classmethod
    def build(
        cls,
        *,
        theta: Array,
        theta_weights: Array,
        phi: Array,
        rho_axis: Array,
        psi_axis: Array,
        z_axis: Array,
        k: float,
        apodization: Literal["none", "sqrt-cos"] = "sqrt-cos",
        mixing: Array | None = None,
        source_pupil: Array | None = None,
        h_cutoff: int | None = None,
        kernel_tolerance: float = 1.0e-12,
        source_tail_tolerance: float = 1.0e-12,
        mode_padding: int = 8,
        miller_margin: int = 32,
        bessel_backend: BesselBackend = "miller",
        allow_underresolved: bool = False,
    ) -> "PreparedVectorDebyeOperator":
        """Prepare one cache and freeze its harmonic-support decision.

        The automatic contract is

        ``H_required = max(H_kernel, H_data) + mode_padding``.

        ``H_data`` is measured after the Richards--Wolf polarization block, so
        its one- and two-mode angular shifts are included.  An even-grid
        Nyquist mode is excluded because it lacks a signed partner.
        """

        theta = _finite_vector("theta", theta)
        theta_weights = _finite_vector("theta_weights", theta_weights)
        phi = _uniform_periodic_phi(phi)
        rho_axis = _finite_vector("rho_axis", rho_axis)
        psi_axis = _finite_vector("psi_axis", psi_axis)
        z_axis = _finite_vector("z_axis", z_axis)
        if theta_weights.shape != theta.shape:
            raise ValueError("theta_weights must match theta")
        if np.any(theta_weights <= 0.0):
            raise ValueError("theta_weights must be positive")
        if np.any(rho_axis < 0.0):
            raise ValueError("rho_axis must be non-negative")
        k = float(k)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("k must be finite and positive")
        if not 0.0 < float(kernel_tolerance) < 1.0:
            raise ValueError("kernel_tolerance must lie in (0, 1)")
        if not 0.0 < float(source_tail_tolerance) < 1.0:
            raise ValueError("source_tail_tolerance must lie in (0, 1)")
        mode_padding = int(mode_padding)
        miller_margin = int(miller_margin)
        if mode_padding < 0 or miller_margin < 0:
            raise ValueError("mode_padding and miller_margin must be non-negative")
        if bessel_backend not in {"miller", "scipy"}:
            raise ValueError("bessel_backend must be 'miller' or 'scipy'")

        if mixing is None:
            mixing_array = richards_wolf_jones_matrix(
                theta, phi, apodization=apodization
            )
        else:
            mixing_array = np.asarray(mixing, dtype=np.complex128)
            if mixing_array.shape != (3, 2, theta.size, phi.size):
                raise ValueError("mixing must have shape (3, 2, ntheta, nphi)")
            if not np.all(np.isfinite(mixing_array)):
                raise ValueError("mixing must contain only finite values")

        x_max = float(k * np.max(rho_axis) * np.max(np.abs(np.sin(theta))))
        kernel_cutoff = estimate_bessel_cutoff(x_max, tol=kernel_tolerance)
        data_cutoff = 0
        if source_pupil is not None:
            effective = mix_jones_pupil(source_pupil, mixing_array)
            data_fourier = np.fft.fft(effective, axis=2)
            data_cutoff = minimum_symmetric_cutoff(
                data_fourier,
                source_tail_tolerance,
                axis=2,
            )
        required_cutoff = max(kernel_cutoff, data_cutoff) + mode_padding
        symmetric_nyquist = (phi.size - 1) // 2
        requested_cutoff = required_cutoff if h_cutoff is None else int(h_cutoff)
        if requested_cutoff < 0:
            raise ValueError("h_cutoff must be non-negative or None")
        underresolved = requested_cutoff < required_cutoff
        nyquist_limited = (
            required_cutoff > symmetric_nyquist
            or requested_cutoff > symmetric_nyquist
        )
        if (underresolved or nyquist_limited) and not allow_underresolved:
            reason = (
                f"requested H={requested_cutoff} is below required H={required_cutoff}"
                if underresolved
                else f"requested/required H={max(requested_cutoff, required_cutoff)} "
                f"exceeds symmetric Nyquist H={symmetric_nyquist}"
            )
            raise ValueError(reason + "; increase n_phi or explicitly allow under-resolution")
        compute_cutoff = min(requested_cutoff, symmetric_nyquist)

        modes = signed_angular_modes(phi.size)
        mode_mask = np.abs(modes) <= compute_cutoff
        h = modes[mode_mask]
        abs_h = np.abs(h)
        unique_abs_h, inverse = np.unique(abs_h, return_inverse=True)
        arguments = (
            k * np.sin(theta)[:, None] * rho_axis[None, :]
        )
        if bessel_backend == "miller":
            bessel_band = cylindrical_bessel_band_miller(
                arguments,
                int(np.max(unique_abs_h, initial=0)),
                margin=miller_margin,
            )
            bessel = bessel_band[:, :, unique_abs_h]
        else:
            bessel = special.jv(
                unique_abs_h[None, None, :], arguments[:, :, None]
            )
        bessel = np.transpose(bessel[:, :, inverse], (0, 2, 1))
        radial = (
            2.0
            * np.pi
            * theta_weights[:, None, None]
            * np.power(1j, abs_h)[None, :, None]
            * bessel
        )
        phi_origin = float(phi[0])
        angular = np.exp(1j * h[:, None] * (psi_axis[None, :] - phi_origin))
        defocus = np.exp(1j * k * np.cos(theta)[:, None] * z_axis[None, :])
        contract = VectorDebyeHarmonicContract(
            n_phi=int(phi.size),
            kernel_cutoff=int(kernel_cutoff),
            data_cutoff=int(data_cutoff),
            mode_padding=mode_padding,
            required_cutoff=int(required_cutoff),
            compute_cutoff=int(compute_cutoff),
            symmetric_nyquist=int(symmetric_nyquist),
            nyquist_limited=bool(nyquist_limited or underresolved),
            kernel_tolerance=float(kernel_tolerance),
            source_tail_tolerance=float(source_tail_tolerance),
            miller_margin=miller_margin,
            bessel_backend=bessel_backend,
        )
        return cls(
            theta=_readonly(theta),
            theta_weights=_readonly(theta_weights),
            phi=_readonly(phi),
            rho_axis=_readonly(rho_axis),
            psi_axis=_readonly(psi_axis),
            z_axis=_readonly(z_axis),
            k=k,
            mixing=_readonly(mixing_array),
            h=_readonly(h),
            mode_mask=_readonly(mode_mask),
            radial=_readonly(radial),
            angular=_readonly(angular),
            defocus=_readonly(defocus),
            contract=contract,
        )

    @property
    def pupil_shape(self) -> tuple[int, int, int]:
        return (2, self.theta.size, self.phi.size)

    @property
    def field_shape(self) -> tuple[int, int]:
        return (3, self.rho_axis.size * self.psi_axis.size * self.z_axis.size)

    @property
    def cache_bytes(self) -> int:
        return int(
            self.mixing.nbytes
            + self.radial.nbytes
            + self.angular.nbytes
            + self.defocus.nbytes
        )

    def forward(self, pupil_jones: Array) -> Array:
        """Apply the prepared vector field operator."""

        pupil = self._validate_pupil(pupil_jones)
        effective = mix_jones_pupil(pupil, self.mixing)
        return self.forward_effective(effective)

    def forward_effective(self, effective_pupil: Array) -> Array:
        """Restrict a prepared three-component angular spectrum.

        This is the coefficient-space boundary between a local polarization or
        Green action and the rotational Fourier restriction.  Exposing it
        makes an unfused primitive chain available for validation while
        :meth:`forward` retains the usual fused Jones-to-field interface.
        """

        effective = self._validate_effective(effective_pupil)
        coeff = np.fft.fft(effective, axis=2)[:, :, self.mode_mask]
        coeff = coeff / float(self.phi.size)
        out = np.empty(self.field_shape, dtype=np.complex128)
        for component in range(3):
            out[component] = self._forward_coeff(coeff[component])
        return out

    def adjoint(self, field_cotangent: Array) -> Array:
        """Apply the Euclidean adjoint paired with :meth:`forward`."""

        effective_gradient = self.adjoint_effective(field_cotangent)
        return unmix_jones_adjoint(effective_gradient, self.mixing)

    def adjoint_effective(self, field_cotangent: Array) -> Array:
        """Adjoint of :meth:`forward_effective` in the sampled Euclidean metric."""

        cotangent = self._validate_field(field_cotangent)
        effective_gradient = np.empty(
            (3, self.theta.size, self.phi.size), dtype=np.complex128
        )
        radial_conj = np.conjugate(self.radial)
        angular_conj = np.conjugate(self.angular)
        defocus_conj = np.conjugate(self.defocus)
        target_shape = (
            self.rho_axis.size,
            self.psi_axis.size,
            self.z_axis.size,
        )
        for component in range(3):
            residual = cotangent[component].reshape(target_shape)
            coeff_gradient = np.einsum(
                "rpz,thr,hp,tz->th",
                residual,
                radial_conj,
                angular_conj,
                defocus_conj,
                optimize=True,
            )
            full = np.zeros(
                (self.theta.size, self.phi.size), dtype=np.complex128
            )
            full[:, self.mode_mask] = coeff_gradient
            effective_gradient[component] = np.fft.ifft(full, axis=1)
        return effective_gradient

    def jvp(self, pupil_direction: Array) -> Array:
        """Jones-space JVP; linearity makes it another forward application."""

        return self.forward(pupil_direction)

    def vjp(self, field_cotangent: Array) -> Array:
        """Jones-space VJP under the Euclidean complex inner product."""

        return self.adjoint(field_cotangent)

    def shared_phase_jvp(self, pupil_jones: Array, phase_direction: Array) -> Array:
        """JVP for one real phase screen shared by both Jones components."""

        pupil = self._validate_pupil(pupil_jones)
        direction = np.asarray(phase_direction, dtype=np.float64)
        if direction.shape != pupil.shape[1:] or not np.all(np.isfinite(direction)):
            raise ValueError("phase_direction must be finite with shape (ntheta, nphi)")
        return self.forward(1j * pupil * direction[None, :, :])

    def shared_phase_vjp(self, pupil_jones: Array, field_cotangent: Array) -> Array:
        """Real VJP for one phase screen shared by both Jones components."""

        pupil = self._validate_pupil(pupil_jones)
        pupil_gradient = self.adjoint(field_cotangent)
        return np.sum(
            np.imag(np.conjugate(pupil) * pupil_gradient), axis=0
        )

    def target_translation_jvp(
        self,
        pupil_jones: Array,
        direction_xyz: Array,
    ) -> Array:
        """Derivative for a common Cartesian translation of all target points."""

        pupil = self._validate_pupil(pupil_jones)
        direction = np.asarray(direction_xyz, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction_xyz must be a finite length-three vector")
        phase_direction = np.tensordot(direction, self._translation_phase_basis(), axes=1)
        return self.shared_phase_jvp(pupil, phase_direction)

    def target_translation_vjp(
        self,
        pupil_jones: Array,
        field_cotangent: Array,
    ) -> Array:
        """Real VJP with respect to a common target translation ``(dx,dy,dz)``."""

        phase_gradient = self.shared_phase_vjp(pupil_jones, field_cotangent)
        return np.einsum(
            "atp,tp->a", self._translation_phase_basis(), phase_gradient, optimize=True
        )

    def translated_pupil(self, pupil_jones: Array, displacement_xyz: Array) -> Array:
        """Apply the exact pupil phase associated with a common target shift."""

        pupil = self._validate_pupil(pupil_jones)
        displacement = np.asarray(displacement_xyz, dtype=np.float64)
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
            raise ValueError("displacement_xyz must be a finite length-three vector")
        phase = np.tensordot(displacement, self._translation_phase_basis(), axes=1)
        return pupil * np.exp(1j * phase)[None, :, :]

    def _forward_coeff(self, coeff: Array) -> Array:
        radial_sum = np.einsum("th,thr->rth", coeff, self.radial, optimize=True)
        angular_sum = np.einsum(
            "rth,hp->rtp", radial_sum, self.angular, optimize=True
        )
        out = np.einsum(
            "rtp,tz->rpz", angular_sum, self.defocus, optimize=True
        )
        return out.ravel()

    def _translation_phase_basis(self) -> Array:
        theta = self.theta[:, None]
        phi = self.phi[None, :]
        return self.k * np.stack(
            np.broadcast_arrays(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta) * np.ones_like(phi),
            ),
            axis=0,
        )

    def _validate_pupil(self, pupil_jones: Array) -> Array:
        pupil = np.asarray(pupil_jones, dtype=np.complex128)
        if pupil.shape != self.pupil_shape:
            raise ValueError(f"pupil_jones must have shape {self.pupil_shape}")
        if not np.all(np.isfinite(pupil)):
            raise ValueError("pupil_jones must contain only finite values")
        return pupil

    def _validate_effective(self, effective_pupil: Array) -> Array:
        effective = np.asarray(effective_pupil, dtype=np.complex128)
        expected = (3, self.theta.size, self.phi.size)
        if effective.shape != expected:
            raise ValueError(f"effective_pupil must have shape {expected}")
        if not np.all(np.isfinite(effective)):
            raise ValueError("effective_pupil must contain only finite values")
        return effective

    def _validate_field(self, field: Array) -> Array:
        values = np.asarray(field, dtype=np.complex128)
        if values.shape != self.field_shape:
            raise ValueError(f"field_cotangent must have shape {self.field_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("field_cotangent must contain only finite values")
        return values


def direct_vector_debye(
    pupil_jones: Array,
    *,
    theta: Array,
    theta_weights: Array,
    phi: Array,
    rho: Array,
    psi: Array,
    z: Array,
    k: float,
    mixing: Array | None = None,
    apodization: Literal["none", "sqrt-cos"] = "sqrt-cos",
) -> Array:
    """Direct sampled Richards--Wolf quadrature for validation."""

    theta = _finite_vector("theta", theta)
    theta_weights = _finite_vector("theta_weights", theta_weights)
    phi = _finite_vector("phi", phi)
    rho = _finite_vector("rho", rho)
    psi = _finite_vector("psi", psi)
    z = _finite_vector("z", z)
    if theta_weights.shape != theta.shape:
        raise ValueError("theta_weights must match theta")
    if not (rho.shape == psi.shape == z.shape):
        raise ValueError("rho, psi, and z must have equal shapes")
    matrix = (
        richards_wolf_jones_matrix(theta, phi, apodization=apodization)
        if mixing is None
        else np.asarray(mixing, dtype=np.complex128)
    )
    effective = mix_jones_pupil(pupil_jones, matrix)
    dphi = 2.0 * np.pi / float(phi.size)
    out = np.zeros((3, rho.size), dtype=np.complex128)
    phi_col = phi[:, None]
    for it, theta_i in enumerate(theta):
        phase = np.exp(
            1j
            * float(k)
            * (
                np.sin(theta_i)
                * rho[None, :]
                * np.cos(phi_col - psi[None, :])
                + np.cos(theta_i) * z[None, :]
            )
        )
        out += theta_weights[it] * dphi * np.einsum(
            "cp,pu->cu", effective[:, it, :], phase, optimize=True
        )
    return out


def cylindrical_bessel_band_miller(
    x: Array, max_order: int, *, margin: int = 32
) -> Array:
    """Return ``J_0(x),...,J_H(x)`` by normalized downward recurrence.

    ``margin`` is a lower bound.  Large arguments receive an additional
    ``4*sqrt(|x|)`` guard because a fixed offset from the turning point loses
    accuracy when ``|x|`` is hundreds or larger.
    """

    values = np.asarray(x, dtype=np.float64)
    max_order = int(max_order)
    margin = int(margin)
    if max_order < 0 or margin < 0:
        raise ValueError("max_order and margin must be non-negative")
    flat = values.ravel()
    out = np.zeros((flat.size, max_order + 1), dtype=np.float64)
    zero = flat == 0.0
    out[zero, 0] = 1.0
    if np.all(zero):
        return out.reshape(values.shape + (max_order + 1,))

    arguments = flat[~zero]
    argument_margins = np.maximum(
        margin,
        np.ceil(4.0 * np.sqrt(np.abs(arguments))).astype(np.int64),
    )
    starts = np.maximum(
        max_order + argument_margins,
        np.ceil(np.abs(arguments)).astype(np.int64) + argument_margins,
    )
    count = arguments.size
    saved = np.zeros((count, max_order + 1), dtype=np.float64)
    b_next = np.zeros(count, dtype=np.float64)
    b_current = np.zeros(count, dtype=np.float64)
    even_tail = np.zeros(count, dtype=np.float64)
    maximum_start = int(np.max(starts))
    for order in range(maximum_start, 0, -1):
        initialize = starts == order
        if np.any(initialize):
            b_current[initialize] = 1.0
            b_next[initialize] = 0.0
            if order <= max_order:
                saved[initialize, order] = 1.0
            if order >= 2 and order % 2 == 0:
                even_tail[initialize] = 2.0

        active = starts >= order
        if not np.any(active):
            continue
        active_arguments = arguments[active]
        previous = (
            (2.0 * order / active_arguments) * b_current[active]
            - b_next[active]
        )
        stored_order = order - 1
        if stored_order <= max_order:
            saved[active, stored_order] = previous
        if stored_order >= 2 and stored_order % 2 == 0:
            even_tail[active] += 2.0 * previous
        b_next[active] = b_current[active]
        b_current[active] = previous

        active_indices = np.flatnonzero(active)
        scale = np.maximum(
            np.abs(b_current[active]), np.abs(b_next[active])
        )
        rescale_local = scale > 1.0e100
        if np.any(rescale_local):
            rescale = active_indices[rescale_local]
            b_current[rescale] *= 1.0e-100
            b_next[rescale] *= 1.0e-100
            even_tail[rescale] *= 1.0e-100
            saved[rescale] *= 1.0e-100

    denominator = saved[:, 0] + even_tail
    if np.any(denominator == 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("Miller recurrence normalization failed")
    out[~zero] = saved / denominator[:, None]
    return out.reshape(values.shape + (max_order + 1,))


def _finite_vector(name: str, values: Array) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return np.array(array, dtype=np.float64, copy=True)


def _uniform_periodic_phi(phi: Array) -> Array:
    values = _finite_vector("phi", phi)
    if values.size < 3:
        raise ValueError("phi must contain at least three samples")
    expected_step = 2.0 * np.pi / float(values.size)
    if not np.allclose(np.diff(values), expected_step, rtol=1.0e-12, atol=1.0e-13):
        raise ValueError("phi must be one complete uniform 2pi orbit")
    return values


def _readonly(values: Array) -> Array:
    array = np.ascontiguousarray(values)
    array.setflags(write=False)
    return array
