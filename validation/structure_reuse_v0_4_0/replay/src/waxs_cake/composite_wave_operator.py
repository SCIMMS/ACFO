"""Task-specific materialization of an SO(2)-equivariant wave-operator chain.

The constructive example in this module combines three independently useful
actions,

``incident channels -> layered dyadic angular spectrum -> rotational
Fourier restriction``,

and materializes their requested finite action on a low-dimensional incident
channel space.  The retained reference chain exposes the same primitives
without materialization, so numerical equivalence, adjoints and geometry
derivatives can be audited independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np

from .harmonic_support import minimum_symmetric_cutoff
from .layered_dyadic_green import isotropic_reflection_dyad
from .vector_debye import (
    PreparedVectorDebyeOperator,
    direct_vector_debye,
    mix_jones_pupil,
    unmix_jones_adjoint,
)


Array = np.ndarray


@dataclass(frozen=True)
class CompositeMaterializationContract:
    """Frozen physical, support and channel contract for one composite."""

    channel_count: int
    channel_labels: tuple[str, ...]
    source_height: float
    lateral_displacement: tuple[float, float]
    upper_wavenumber: float
    lower_wavenumber: float
    damping: float
    channel_data_cutoff: int
    active_harmonic_cutoff: int
    mode_padding: int
    miller_margin: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def layered_reflected_jones_green_mixing(
    theta: Any,
    phi: Any,
    *,
    upper_wavenumber: float,
    lower_wavenumber: float,
    damping: float,
    source_height: float,
    lateral_displacement: tuple[float, float] = (0.0, 0.0),
) -> Array:
    r"""Return a reflected dyadic Green block in sampled angular coordinates.

    The output has shape ``(field_component, Jones_component, theta, phi)``.
    The Sommerfeld radial measure is converted from ``q dq / kz`` to the
    ``sin(theta) dtheta`` measure used by :class:`PreparedVectorDebyeOperator`.
    Only the two transverse source components are exposed as incident Jones
    channels; the returned field retains all three Cartesian components.
    """

    theta_values = _finite_vector("theta", theta)
    phi_values = _finite_vector("phi", phi)
    upper = float(upper_wavenumber)
    lower = float(lower_wavenumber)
    damping_value = float(damping)
    height = float(source_height)
    dx, dy = (float(lateral_displacement[0]), float(lateral_displacement[1]))
    if (
        not np.isfinite(upper)
        or not np.isfinite(lower)
        or not np.isfinite(damping_value)
        or not np.isfinite(height)
        or not np.isfinite(dx)
        or not np.isfinite(dy)
        or upper <= 0.0
        or lower <= 0.0
        or damping_value <= 0.0
        or height <= 0.0
    ):
        raise ValueError(
            "wavenumbers, damping and source_height must be finite and positive; "
            "lateral_displacement must be finite"
        )
    if np.any(theta_values < 0.0) or np.any(theta_values >= 0.5 * np.pi):
        raise ValueError("theta must lie in [0, pi/2) for the reflected angular map")

    radial = upper * np.sin(theta_values)
    dyad, axial, _, _ = isotropic_reflection_dyad(
        radial,
        phi_values,
        upper_wavenumber=upper,
        lower_wavenumber=lower,
        damping=damping_value,
    )
    jacobian = upper**2 * np.cos(theta_values) / axial
    propagation = np.exp(1j * axial * height)
    qx = radial[:, None] * np.cos(phi_values)[None, :]
    qy = radial[:, None] * np.sin(phi_values)[None, :]
    lateral_phase = np.exp(1j * (qx * dx + qy * dy))
    scale = (
        1j
        / (8.0 * np.pi**2)
        * jacobian[:, None]
        * propagation[:, None]
        * lateral_phase
    )
    # isotropic_reflection_dyad stores (theta, phi, field, source).
    mixing = dyad[..., :, :2] * scale[..., None, None]
    return np.transpose(mixing, (2, 3, 0, 1)).astype(
        np.complex128, copy=False
    )


@dataclass(frozen=True)
class MaterializedSO2ChannelOperator:
    """Compact deployment form containing only prepared finite actions."""

    operator_matrix: Array
    lateral_derivative_matrices: Array
    contract: CompositeMaterializationContract

    @property
    def channel_shape(self) -> tuple[int]:
        return (self.contract.channel_count,)

    @property
    def field_shape(self) -> tuple[int, int]:
        return tuple(self.operator_matrix.shape[:2])  # type: ignore[return-value]

    @property
    def cache_bytes(self) -> int:
        return int(
            self.operator_matrix.nbytes + self.lateral_derivative_matrices.nbytes
        )

    def forward(self, channel_coefficients: Any) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        return np.einsum(
            "iub,b->iu", self.operator_matrix, coefficients, optimize=True
        )

    def adjoint(self, field_cotangent: Any) -> Array:
        cotangent = self._validate_field(field_cotangent)
        return np.einsum(
            "iub,iu->b",
            np.conjugate(self.operator_matrix),
            cotangent,
            optimize=True,
        )

    def lateral_translation_jvp(
        self, channel_coefficients: Any, direction_xy: Any
    ) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        direction = _validate_lateral_direction(direction_xy)
        return np.einsum(
            "aiub,a,b->iu",
            self.lateral_derivative_matrices,
            direction,
            coefficients,
            optimize=True,
        )

    def lateral_translation_vjp(
        self, channel_coefficients: Any, field_cotangent: Any
    ) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        cotangent = self._validate_field(field_cotangent)
        derivative_fields = np.einsum(
            "aiub,b->aiu",
            self.lateral_derivative_matrices,
            coefficients,
            optimize=True,
        )
        return np.real(
            np.einsum(
                "aiu,iu->a",
                np.conjugate(derivative_fields),
                cotangent,
                optimize=True,
            )
        )

    def normal_matvec(self, channel_coefficients: Any, weights: Any = 1.0) -> Array:
        field = self.forward(channel_coefficients)
        weight_array = np.asarray(weights)
        try:
            weighted = field * np.broadcast_to(weight_array, self.field_shape)
        except ValueError as exc:
            raise ValueError("weights must broadcast to field_shape") from exc
        if not np.all(np.isfinite(weighted)):
            raise ValueError("weights must produce finite weighted fields")
        return self.adjoint(weighted)

    def _validate_channels(self, channel_coefficients: Any) -> Array:
        coefficients = np.asarray(channel_coefficients, dtype=np.complex128)
        if coefficients.shape != self.channel_shape:
            raise ValueError(
                f"channel_coefficients must have shape {self.channel_shape}"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("channel_coefficients must be finite")
        return coefficients

    def _validate_field(self, field: Any) -> Array:
        values = np.asarray(field, dtype=np.complex128)
        if values.shape != self.field_shape:
            raise ValueError(f"field must have shape {self.field_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("field must be finite")
        return values


@dataclass(frozen=True)
class PreparedLayeredVectorCompositeOperator:
    r"""Materialized ``A_Gamma G_layer T(d)`` on incident channels.

    ``source_basis[b]`` is one sampled incident Jones channel.  Preparation
    applies the layered dyadic Green block and the rotational restriction to
    each channel once, storing a compact channel-to-observable matrix and its
    two lateral-translation derivatives.  Repeated forward, adjoint, JVP/VJP
    and normal actions then use only these matrices.
    """

    source_basis: Array
    restriction: PreparedVectorDebyeOperator
    operator_matrix: Array
    lateral_derivative_matrices: Array
    lateral_phase_basis: Array
    contract: CompositeMaterializationContract

    @classmethod
    def build(
        cls,
        *,
        theta: Any,
        theta_weights: Any,
        phi: Any,
        rho_axis: Any,
        psi_axis: Any,
        z_axis: Any,
        upper_wavenumber: float,
        lower_wavenumber: float,
        damping: float,
        source_height: float,
        source_basis: Any,
        channel_labels: Sequence[str] | None = None,
        lateral_displacement: tuple[float, float] = (0.0, 0.0),
        kernel_tolerance: float = 1.0e-12,
        source_tail_tolerance: float = 1.0e-12,
        mode_padding: int = 8,
        miller_margin: int = 32,
        bessel_backend: str = "miller",
    ) -> "PreparedLayeredVectorCompositeOperator":
        theta_values = _finite_vector("theta", theta)
        phi_values = _finite_vector("phi", phi)
        basis = np.asarray(source_basis, dtype=np.complex128)
        expected_tail = (2, theta_values.size, phi_values.size)
        if basis.ndim != 4 or tuple(basis.shape[1:]) != expected_tail:
            raise ValueError(
                "source_basis must have shape (n_channel, 2, ntheta, nphi)"
            )
        if basis.shape[0] == 0 or not np.all(np.isfinite(basis)):
            raise ValueError("source_basis must contain finite non-empty channels")
        if channel_labels is None:
            labels = tuple(f"channel_{index}" for index in range(basis.shape[0]))
        else:
            labels = tuple(str(value) for value in channel_labels)
            if len(labels) != basis.shape[0] or any(not label for label in labels):
                raise ValueError("channel_labels must name every source channel")

        mixing = layered_reflected_jones_green_mixing(
            theta_values,
            phi_values,
            upper_wavenumber=upper_wavenumber,
            lower_wavenumber=lower_wavenumber,
            damping=damping,
            source_height=source_height,
            lateral_displacement=lateral_displacement,
        )
        effective_channels = np.stack(
            [mix_jones_pupil(channel, mixing) for channel in basis], axis=0
        )
        channel_cutoffs = np.asarray(
            [
                minimum_symmetric_cutoff(
                    np.fft.fft(effective, axis=2),
                    source_tail_tolerance,
                    axis=2,
                )
                for effective in effective_channels
            ],
            dtype=np.int64,
        )
        support_index = int(np.argmax(channel_cutoffs))
        restriction = PreparedVectorDebyeOperator.build(
            theta=theta_values,
            theta_weights=theta_weights,
            phi=phi_values,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            k=upper_wavenumber,
            mixing=mixing,
            source_pupil=basis[support_index],
            kernel_tolerance=kernel_tolerance,
            source_tail_tolerance=source_tail_tolerance,
            mode_padding=mode_padding,
            miller_margin=miller_margin,
            bessel_backend=bessel_backend,
        )
        matrix = np.stack(
            [restriction.forward(channel) for channel in basis], axis=-1
        )
        radial = float(upper_wavenumber) * np.sin(theta_values)[:, None]
        lateral_phase_basis = np.stack(
            (
                radial * np.cos(phi_values)[None, :],
                radial * np.sin(phi_values)[None, :],
            ),
            axis=0,
        )
        derivative_matrices = np.empty(
            (2,) + matrix.shape, dtype=np.complex128
        )
        for axis in range(2):
            derivative_matrices[axis] = np.stack(
                [
                    restriction.shared_phase_jvp(
                        channel, lateral_phase_basis[axis]
                    )
                    for channel in basis
                ],
                axis=-1,
            )
        contract = CompositeMaterializationContract(
            channel_count=int(basis.shape[0]),
            channel_labels=labels,
            source_height=float(source_height),
            lateral_displacement=(
                float(lateral_displacement[0]),
                float(lateral_displacement[1]),
            ),
            upper_wavenumber=float(upper_wavenumber),
            lower_wavenumber=float(lower_wavenumber),
            damping=float(damping),
            channel_data_cutoff=int(np.max(channel_cutoffs)),
            active_harmonic_cutoff=int(restriction.contract.compute_cutoff),
            mode_padding=int(mode_padding),
            miller_margin=int(miller_margin),
        )
        return cls(
            source_basis=_readonly(basis),
            restriction=restriction,
            operator_matrix=_readonly(matrix),
            lateral_derivative_matrices=_readonly(derivative_matrices),
            lateral_phase_basis=_readonly(lateral_phase_basis),
            contract=contract,
        )

    @property
    def channel_shape(self) -> tuple[int]:
        return (self.contract.channel_count,)

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.restriction.field_shape

    @property
    def materialized_bytes(self) -> int:
        return int(
            self.operator_matrix.nbytes + self.lateral_derivative_matrices.nbytes
        )

    @property
    def reference_chain_bytes(self) -> int:
        return int(self.source_basis.nbytes + self.restriction.cache_bytes)

    @property
    def avoided_intermediate_bytes_per_call(self) -> int:
        return int(
            (2 + 3)
            * self.restriction.theta.size
            * self.restriction.phi.size
            * np.dtype(np.complex128).itemsize
        )

    def compact(self) -> MaterializedSO2ChannelOperator:
        """Return the deployable action without the retained audit chain."""

        return MaterializedSO2ChannelOperator(
            operator_matrix=self.operator_matrix,
            lateral_derivative_matrices=self.lateral_derivative_matrices,
            contract=self.contract,
        )

    def materialize_lateral_update(
        self, displacement_xy: Any
    ) -> MaterializedSO2ChannelOperator:
        """Materialize an exact finite lateral update from the retained primitives."""

        displacement = self._validate_lateral_direction(displacement_xy)
        phase = np.einsum(
            "a,atp->tp", displacement, self.lateral_phase_basis, optimize=True
        )
        translated_basis = self.source_basis * np.exp(1j * phase)[None, None, :, :]
        matrix = np.stack(
            [self.restriction.forward(channel) for channel in translated_basis],
            axis=-1,
        )
        derivatives = np.empty((2,) + matrix.shape, dtype=np.complex128)
        for axis in range(2):
            derivatives[axis] = np.stack(
                [
                    self.restriction.shared_phase_jvp(
                        channel, self.lateral_phase_basis[axis]
                    )
                    for channel in translated_basis
                ],
                axis=-1,
            )
        anchor = np.asarray(self.contract.lateral_displacement, dtype=np.float64)
        contract = replace(
            self.contract,
            lateral_displacement=tuple(anchor + displacement),
        )
        return MaterializedSO2ChannelOperator(
            operator_matrix=_readonly(matrix),
            lateral_derivative_matrices=_readonly(derivatives),
            contract=contract,
        )

    def forward(self, channel_coefficients: Any) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        return np.einsum(
            "iub,b->iu", self.operator_matrix, coefficients, optimize=True
        )

    def adjoint(self, field_cotangent: Any) -> Array:
        cotangent = self._validate_field(field_cotangent)
        return np.einsum(
            "iub,iu->b",
            np.conjugate(self.operator_matrix),
            cotangent,
            optimize=True,
        )

    def lateral_translation_jvp(
        self, channel_coefficients: Any, direction_xy: Any
    ) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        direction = self._validate_lateral_direction(direction_xy)
        return np.einsum(
            "aiub,a,b->iu",
            self.lateral_derivative_matrices,
            direction,
            coefficients,
            optimize=True,
        )

    def lateral_translation_vjp(
        self, channel_coefficients: Any, field_cotangent: Any
    ) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        cotangent = self._validate_field(field_cotangent)
        derivative_fields = np.einsum(
            "aiub,b->aiu",
            self.lateral_derivative_matrices,
            coefficients,
            optimize=True,
        )
        return np.real(
            np.einsum(
                "aiu,iu->a",
                np.conjugate(derivative_fields),
                cotangent,
                optimize=True,
            )
        )

    def normal_matvec(self, channel_coefficients: Any, weights: Any = 1.0) -> Array:
        field = self.forward(channel_coefficients)
        weight_array = np.asarray(weights)
        try:
            weighted = field * np.broadcast_to(weight_array, self.field_shape)
        except ValueError as exc:
            raise ValueError("weights must broadcast to field_shape") from exc
        if not np.all(np.isfinite(weighted)):
            raise ValueError("weights must produce finite weighted fields")
        return self.adjoint(weighted)

    def reference_forward(self, channel_coefficients: Any) -> Array:
        pupil = self._channel_pupil(channel_coefficients)
        effective = mix_jones_pupil(pupil, self.restriction.mixing)
        return self.restriction.forward_effective(effective)

    def reference_adjoint(self, field_cotangent: Any) -> Array:
        cotangent = self._validate_field(field_cotangent)
        effective_gradient = self.restriction.adjoint_effective(cotangent)
        pupil_gradient = unmix_jones_adjoint(
            effective_gradient, self.restriction.mixing
        )
        return np.einsum(
            "bjtp,jtp->b",
            np.conjugate(self.source_basis),
            pupil_gradient,
            optimize=True,
        )

    def reference_lateral_translation_jvp(
        self, channel_coefficients: Any, direction_xy: Any
    ) -> Array:
        pupil = self._channel_pupil(channel_coefficients)
        direction = self._validate_lateral_direction(direction_xy)
        phase_direction = np.einsum(
            "a,atp->tp", direction, self.lateral_phase_basis, optimize=True
        )
        return self.restriction.shared_phase_jvp(pupil, phase_direction)

    def reference_translated_forward(
        self, channel_coefficients: Any, displacement_xy: Any
    ) -> Array:
        pupil = self._channel_pupil(channel_coefficients)
        displacement = self._validate_lateral_direction(displacement_xy)
        phase = np.einsum(
            "a,atp->tp", displacement, self.lateral_phase_basis, optimize=True
        )
        translated = pupil * np.exp(1j * phase)[None, :, :]
        return self.restriction.forward(translated)

    def direct_forward(self, channel_coefficients: Any) -> Array:
        pupil = self._channel_pupil(channel_coefficients)
        rr, pp, zz = np.meshgrid(
            self.restriction.rho_axis,
            self.restriction.psi_axis,
            self.restriction.z_axis,
            indexing="ij",
        )
        return direct_vector_debye(
            pupil,
            theta=self.restriction.theta,
            theta_weights=self.restriction.theta_weights,
            phi=self.restriction.phi,
            rho=rr.ravel(),
            psi=pp.ravel(),
            z=zz.ravel(),
            k=self.restriction.k,
            mixing=self.restriction.mixing,
        )

    def _channel_pupil(self, channel_coefficients: Any) -> Array:
        coefficients = self._validate_channels(channel_coefficients)
        return np.einsum(
            "b,bjtp->jtp", coefficients, self.source_basis, optimize=True
        )

    def _validate_channels(self, channel_coefficients: Any) -> Array:
        coefficients = np.asarray(channel_coefficients, dtype=np.complex128)
        if coefficients.shape != self.channel_shape:
            raise ValueError(
                f"channel_coefficients must have shape {self.channel_shape}"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("channel_coefficients must be finite")
        return coefficients

    def _validate_field(self, field: Any) -> Array:
        values = np.asarray(field, dtype=np.complex128)
        if values.shape != self.field_shape:
            raise ValueError(f"field must have shape {self.field_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("field must be finite")
        return values

    @staticmethod
    def _validate_lateral_direction(direction_xy: Any) -> Array:
        return _validate_lateral_direction(direction_xy)


def _finite_vector(name: str, values: Any) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _validate_lateral_direction(direction_xy: Any) -> Array:
    direction = np.asarray(direction_xy, dtype=np.float64)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("lateral direction must be a finite length-two vector")
    return direction


def _readonly(values: Array) -> Array:
    array = np.ascontiguousarray(values)
    array.setflags(write=False)
    return array
