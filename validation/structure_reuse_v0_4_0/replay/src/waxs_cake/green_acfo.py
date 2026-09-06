"""Spectral Green-operator actions composed from ACFO forward and adjoint."""

from __future__ import annotations

from typing import Any


class AxisymmetricSpectralGreenOperator:
    """Apply axisymmetric Fourier multipliers with a paired ACFO transform.

    ``meridional_weights`` approximates ``q_perp dq_perp dq_z`` at each
    meridional target.  Uniform angular quadrature and the ``(2*pi)^-3``
    inverse-Fourier normalization are applied internally.
    """

    def __init__(self, operator: Any, meridional_weights: Any) -> None:
        self.operator = operator
        self.torch = operator.torch
        self.device = operator.device
        weights = self.torch.as_tensor(
            meridional_weights,
            dtype=operator.real_dtype,
            device=operator.device,
        )
        if tuple(weights.shape) != (operator.n_q,) or not bool(
            self.torch.all(self.torch.isfinite(weights)).item()
        ):
            raise ValueError(
                f"meridional_weights must be finite with shape ({operator.n_q},)"
            )
        if bool(self.torch.any(weights < 0.0).item()):
            raise ValueError("meridional_weights must be non-negative")
        delta_phi = 2.0 * self.torch.pi / operator.n_phi
        self.inverse_fourier_weights = weights * delta_phi / (2.0 * self.torch.pi) ** 3

    def _multiplier(self, values: Any) -> Any:
        multiplier = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.operator.complex_dtype
        )
        if tuple(multiplier.shape) != (self.operator.n_q,) or not bool(
            self.torch.all(self.torch.isfinite(multiplier)).item()
        ):
            raise ValueError(
                f"spectral multiplier must be finite with shape ({self.operator.n_q},)"
            )
        return multiplier

    def apply(self, source_values: Any, spectral_multiplier: Any) -> Any:
        spectrum = self.operator.forward(source_values)
        return self._apply_spectrum(spectrum, spectral_multiplier)

    def _apply_spectrum(self, spectrum: Any, spectral_multiplier: Any) -> Any:
        multiplier = self._multiplier(spectral_multiplier)
        weighted = spectrum * (
            self.inverse_fourier_weights.to(dtype=self.operator.complex_dtype)
            * multiplier
        )[:, None]
        return self.operator.adjoint(weighted)

    def yukawa_multiplier(self, kappa: float | Any) -> Any:
        kappa_tensor = self.torch.as_tensor(
            kappa, dtype=self.operator.real_dtype, device=self.device
        )
        if kappa_tensor.ndim != 0 or not bool(self.torch.isfinite(kappa_tensor).item()):
            raise ValueError("kappa must be a finite scalar")
        if float(kappa_tensor.detach().cpu()) <= 0.0:
            raise ValueError("kappa must be positive")
        q_squared = self.operator.default_q_perp**2 + self.operator.default_q_z**2
        return 1.0 / (q_squared + kappa_tensor**2)

    def apply_yukawa(self, source_values: Any, kappa: float | Any) -> Any:
        return self.apply(source_values, self.yukawa_multiplier(kappa))

    def complex_helmholtz_multiplier(
        self, wavenumber: float | Any, damping: float | Any
    ) -> Any:
        k = self.torch.as_tensor(
            wavenumber, dtype=self.operator.real_dtype, device=self.device
        )
        eta = self.torch.as_tensor(
            damping, dtype=self.operator.real_dtype, device=self.device
        )
        if k.ndim != 0 or eta.ndim != 0 or not bool(
            self.torch.isfinite(k).item() and self.torch.isfinite(eta).item()
        ):
            raise ValueError("wavenumber and damping must be finite scalars")
        if float(k.detach().cpu()) <= 0.0 or float(eta.detach().cpu()) <= 0.0:
            raise ValueError("wavenumber and damping must be positive")
        q_squared = self.operator.default_q_perp**2 + self.operator.default_q_z**2
        complex_k = k.to(dtype=self.operator.complex_dtype) + 1j * eta.to(
            dtype=self.operator.complex_dtype
        )
        return 1.0 / (q_squared.to(dtype=self.operator.complex_dtype) - complex_k**2)

    def apply_complex_helmholtz(
        self, source_values: Any, wavenumber: float | Any, damping: float | Any
    ) -> Any:
        return self.apply(
            source_values,
            self.complex_helmholtz_multiplier(wavenumber, damping),
        )

    def uniaxial_helmholtz_multiplier(
        self,
        transverse_coefficient: float | Any,
        axial_coefficient: float | Any,
        wavenumber: float | Any,
        damping: float | Any,
    ) -> Any:
        transverse = self.torch.as_tensor(
            transverse_coefficient, dtype=self.operator.real_dtype, device=self.device
        )
        axial = self.torch.as_tensor(
            axial_coefficient, dtype=self.operator.real_dtype, device=self.device
        )
        if transverse.ndim != 0 or axial.ndim != 0 or not bool(
            self.torch.isfinite(transverse).item()
            and self.torch.isfinite(axial).item()
        ):
            raise ValueError("uniaxial coefficients must be finite scalars")
        if float(transverse.detach().cpu()) <= 0.0 or float(axial.detach().cpu()) <= 0.0:
            raise ValueError("uniaxial coefficients must be positive")
        base = self.complex_helmholtz_multiplier(wavenumber, damping)
        complex_k_squared = (
            self.operator.default_q_perp**2
            + self.operator.default_q_z**2
            - 1.0 / base
        )
        denominator = (
            transverse * self.operator.default_q_perp**2
            + axial * self.operator.default_q_z**2
            - complex_k_squared
        )
        return 1.0 / denominator.to(dtype=self.operator.complex_dtype)

    def apply_uniaxial_helmholtz(
        self,
        source_values: Any,
        transverse_coefficient: float | Any,
        axial_coefficient: float | Any,
        wavenumber: float | Any,
        damping: float | Any,
    ) -> Any:
        return self.apply(
            source_values,
            self.uniaxial_helmholtz_multiplier(
                transverse_coefficient,
                axial_coefficient,
                wavenumber,
                damping,
            ),
        )

    def resolvent_multiplier_jet(
        self,
        wavenumber: float | Any,
        damping: float | Any,
        max_order: int,
    ) -> tuple[Any, ...]:
        """Return normalized derivatives with respect to ``lambda = k_c^2``.

        For ``R(lambda) = (q^2-lambda)^-1``, the normalized p-th
        derivative is ``(q^2-lambda)^(-p-1)``.
        """

        max_order = int(max_order)
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        base = self.complex_helmholtz_multiplier(wavenumber, damping)
        return tuple(base ** (order + 1) for order in range(max_order + 1))

    def apply_resolvent_jet(
        self,
        source_values: Any,
        wavenumber: float | Any,
        damping: float | Any,
        max_order: int,
    ) -> tuple[Any, ...]:
        """Apply a frequency-resolvent jet while reusing one source transform."""

        spectrum = self.operator.forward(source_values)
        return tuple(
            self._apply_spectrum(spectrum, multiplier)
            for multiplier in self.resolvent_multiplier_jet(
                wavenumber, damping, max_order
            )
        )

    def evaluate_resolvent_jet(
        self, field_jet: tuple[Any, ...] | list[Any], delta_lambda: Any
    ) -> Any:
        if not field_jet:
            raise ValueError("field_jet must contain at least one coefficient")
        delta = self.torch.as_tensor(
            delta_lambda, dtype=self.operator.complex_dtype, device=self.device
        )
        if delta.ndim != 0 or not bool(self.torch.isfinite(delta).item()):
            raise ValueError("delta_lambda must be a finite scalar")
        output = self.torch.zeros_like(field_jet[0])
        for order, coefficient in enumerate(field_jet):
            output = output + coefficient * delta**order
        return output


class AxisymmetricLippmannSchwingerSolver:
    """Fixed-point Lippmann--Schwinger iteration using a spectral Green action."""

    def __init__(
        self,
        green_operator: AxisymmetricSpectralGreenOperator,
        spectral_multiplier: Any,
        susceptibility: Any,
        spatial_weights: Any,
    ) -> None:
        self.green_operator = green_operator
        self.operator = green_operator.operator
        self.torch = self.operator.torch
        self.multiplier = green_operator._multiplier(spectral_multiplier)
        self.susceptibility = self.torch.as_tensor(
            susceptibility, device=self.operator.device
        ).to(dtype=self.operator.complex_dtype)
        self.spatial_weights = self.torch.as_tensor(
            spatial_weights,
            dtype=self.operator.real_dtype,
            device=self.operator.device,
        )
        if tuple(self.susceptibility.shape) != self.operator.object_shape:
            raise ValueError(
                f"susceptibility must have shape {self.operator.object_shape}"
            )
        if tuple(self.spatial_weights.shape) != self.operator.object_shape:
            raise ValueError(
                f"spatial_weights must have shape {self.operator.object_shape}"
            )
        if not bool(self.torch.all(self.torch.isfinite(self.susceptibility)).item()):
            raise ValueError("susceptibility must be finite")
        if not bool(self.torch.all(self.torch.isfinite(self.spatial_weights)).item()) or bool(
            self.torch.any(self.spatial_weights < 0.0).item()
        ):
            raise ValueError("spatial_weights must be finite and non-negative")

    def interaction(self, field: Any) -> Any:
        value = self.operator._object_tensor(field)
        source = value * self.susceptibility * self.spatial_weights
        return self.green_operator.apply(source, self.multiplier)

    def solve(
        self,
        incident_field: Any,
        *,
        maximum_iterations: int = 100,
        relative_tolerance: float = 1e-10,
        relaxation: float = 1.0,
    ) -> dict[str, Any]:
        incident = self.operator._object_tensor(incident_field)
        maximum_iterations = int(maximum_iterations)
        relative_tolerance = float(relative_tolerance)
        relaxation = float(relaxation)
        if maximum_iterations <= 0 or relative_tolerance <= 0.0:
            raise ValueError("iteration count and tolerance must be positive")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0,1]")
        field = incident.clone()
        denominator = self.torch.linalg.vector_norm(incident).clamp_min(1e-300)
        history = []
        converged = False
        for iteration in range(1, maximum_iterations + 1):
            fixed_point = incident + self.interaction(field)
            updated = (1.0 - relaxation) * field + relaxation * fixed_point
            relative_update = self.torch.linalg.vector_norm(updated - field) / denominator
            history.append(float(relative_update.detach().cpu()))
            field = updated
            if history[-1] <= relative_tolerance:
                converged = True
                break
        residual = field - incident - self.interaction(field)
        relative_residual = self.torch.linalg.vector_norm(residual) / denominator
        return {
            "field": field,
            "converged": converged,
            "iterations": len(history),
            "relative_update_history": history,
            "relative_residual": float(relative_residual.detach().cpu()),
        }
