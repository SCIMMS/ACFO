"""Sommerfeld reflection from a finite uniaxial or hyperbolic slab."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy import special

from .sommerfeld_acfo import ScalarPlanarHalfspaceSommerfeldOperator


Branch = Literal["ordinary", "extraordinary"]


class ScalarUniaxialSlabSommerfeldOperator(
    ScalarPlanarHalfspaceSommerfeldOperator
):
    r"""Evaluate a scalar TE/TM branch reflected by an optic-axis-normal slab.

    The ordinary branch uses

    .. math:: k_{zs}^2=\epsilon_\perp k_0^2-k_\rho^2

    with TE flux admittance ``Y=k_z``.  The extraordinary branch uses

    .. math:: k_{zs}^2=\epsilon_\perp k_0^2
       -(\epsilon_\perp/\epsilon_\parallel)k_\rho^2

    and the TM scalar admittance ``Y=k_z/epsilon_perpendicular``.  The same
    isotropic ambient is used above and below the slab.  Complex passive
    permittivities allow elliptic and hyperbolic regimes while the outgoing
    square-root branch keeps decaying evanescent fields.
    """

    def __init__(
        self,
        radial_wavenumbers: Any,
        quadrature_weights: Any,
        vacuum_wavenumber: float,
        ambient_permittivity: float,
        epsilon_perpendicular: complex,
        epsilon_parallel: complex,
        slab_thickness: float,
        damping: float,
        branch: Branch,
        *,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
    ) -> None:
        vacuum_wavenumber = float(vacuum_wavenumber)
        ambient_permittivity = float(ambient_permittivity)
        slab_thickness = float(slab_thickness)
        damping = float(damping)
        eps_perp = complex(epsilon_perpendicular)
        eps_parallel = complex(epsilon_parallel)
        if branch not in ("ordinary", "extraordinary"):
            raise ValueError("branch must be ordinary or extraordinary")
        if (
            not np.isfinite(vacuum_wavenumber)
            or not np.isfinite(ambient_permittivity)
            or not np.isfinite(slab_thickness)
            or not np.isfinite(damping)
            or vacuum_wavenumber <= 0.0
            or ambient_permittivity <= 0.0
            or slab_thickness <= 0.0
            or damping <= 0.0
        ):
            raise ValueError(
                "wavenumber, ambient permittivity, thickness and damping must be positive"
            )
        if not (
            np.isfinite(eps_perp.real)
            and np.isfinite(eps_perp.imag)
            and np.isfinite(eps_parallel.real)
            and np.isfinite(eps_parallel.imag)
        ):
            raise ValueError("slab permittivities must be finite")
        if eps_perp.imag < 0.0 or eps_parallel.imag < 0.0:
            raise ValueError("slab permittivities must be passive")
        if abs(eps_perp) == 0.0 or abs(eps_parallel) == 0.0:
            raise ValueError("slab permittivities must be nonzero")

        ambient_scale = math.sqrt(ambient_permittivity)
        super().__init__(
            radial_wavenumbers,
            quadrature_weights,
            ambient_scale * vacuum_wavenumber,
            ambient_scale * vacuum_wavenumber,
            ambient_scale * damping,
            torch=torch,
            device=device,
            complex_dtype=complex_dtype,
        )
        self.vacuum_wavenumber = vacuum_wavenumber
        self.ambient_permittivity = ambient_permittivity
        self.epsilon_perpendicular = eps_perp
        self.epsilon_parallel = eps_parallel
        self.slab_thickness = slab_thickness
        self.branch = branch

        complex_frequency = self.torch.as_tensor(
            complex(vacuum_wavenumber, damping),
            dtype=self.complex_dtype,
            device=self.device,
        )
        q = self.radial_wavenumbers.to(dtype=self.complex_dtype)
        eps_t = self.torch.as_tensor(
            eps_perp, dtype=self.complex_dtype, device=self.device
        )
        eps_z = self.torch.as_tensor(
            eps_parallel, dtype=self.complex_dtype, device=self.device
        )
        if branch == "ordinary":
            slab_argument = eps_t * complex_frequency**2 - q**2
            ambient_flux = self.axial_upper
            slab_flux_coefficient = self.torch.ones_like(eps_t)
        else:
            slab_argument = (
                eps_t * complex_frequency**2 - (eps_t / eps_z) * q**2
            )
            ambient_flux = self.axial_upper / ambient_permittivity
            slab_flux_coefficient = 1.0 / eps_t
        self.axial_slab = self._outgoing_root(slab_argument)
        slab_flux = slab_flux_coefficient * self.axial_slab
        interface_reflection = (ambient_flux - slab_flux) / (
            ambient_flux + slab_flux
        )
        round_trip = self.torch.exp(2j * self.axial_slab * slab_thickness)
        denominator = 1.0 - interface_reflection**2 * round_trip
        reflection = interface_reflection * (1.0 - round_trip) / denominator
        derivative_reflection = (
            2j
            * self.axial_slab
            * round_trip
            * interface_reflection
            * (interface_reflection**2 - 1.0)
            / denominator**2
        )
        prefactor = (
            1j
            / (4.0 * math.pi)
            * self.quadrature_weights.to(dtype=self.complex_dtype)
            * q
            / self.axial_upper
        )
        self.interface_reflection = interface_reflection
        self.round_trip = round_trip
        self.reflection = reflection
        self.reflection_thickness_derivative = derivative_reflection
        self.spectral_weights = prefactor * reflection
        self.thickness_derivative_spectral_weights = (
            prefactor * derivative_reflection
        )
        self.prepared_mib = float(
            sum(
                tensor.nelement() * tensor.element_size()
                for tensor in (
                    self.radial_wavenumbers,
                    self.quadrature_weights,
                    self.axial_upper,
                    self.axial_slab,
                    self.interface_reflection,
                    self.round_trip,
                    self.reflection,
                    self.reflection_thickness_derivative,
                    self.spectral_weights,
                    self.thickness_derivative_spectral_weights,
                )
            )
            / (1024.0 * 1024.0)
        )

    def evaluate_thickness_jvp(
        self, rho: Any, height: Any, delta_thickness: Any = 1.0
    ) -> Any:
        """Apply the analytic derivative with respect to slab thickness."""

        radial, axial, shape = self._geometry(rho, height)
        delta = self.torch.as_tensor(
            delta_thickness, dtype=self.real_dtype, device=self.device
        )
        if delta.ndim != 0 or not bool(self.torch.isfinite(delta).item()):
            raise ValueError("delta_thickness must be a finite scalar")
        argument = (
            radial.detach().cpu().numpy()[:, None]
            * self.radial_wavenumbers.detach().cpu().numpy()[None, :]
        )
        bessel = self.torch.as_tensor(
            np.ascontiguousarray(special.jv(0, argument)),
            dtype=self.real_dtype,
            device=self.device,
        )
        phase = self.torch.exp(
            1j * axial[:, None] * self.axial_upper[None, :]
        )
        value = self.torch.sum(
            phase
            * bessel
            * self.thickness_derivative_spectral_weights[None, :],
            dim=-1,
        )
        return (delta * value).reshape(shape)
