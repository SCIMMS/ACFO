"""SO(2) Sommerfeld Green evaluation with analytic geometry derivatives.

The implementation covers the reflected scalar Helmholtz Green function for a
single planar interface.  It is deliberately a small first validation step:
the polarization-dependent dyadic weights required by Maxwell's equations are
outside this module.  A positive complex shift regularizes the radiation
limit, while the outgoing square-root branch retains propagating and
evanescent spectral components.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import special


def segmented_legendre_quadrature(
    breakpoints: Any,
    nodes_per_segment: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes and weights on adjacent finite segments."""

    edges = np.asarray(breakpoints, dtype=np.float64)
    nodes_per_segment = int(nodes_per_segment)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or edges[0] < 0.0
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("breakpoints must be finite, increasing and non-negative")
    if nodes_per_segment <= 0:
        raise ValueError("nodes_per_segment must be positive")
    canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(
        nodes_per_segment
    )
    nodes = []
    weights = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        scale = 0.5 * (upper - lower)
        center = 0.5 * (upper + lower)
        nodes.append(scale * canonical_nodes + center)
        weights.append(scale * canonical_weights)
    return np.concatenate(nodes), np.concatenate(weights)


class ScalarPlanarHalfspaceSommerfeldOperator:
    r"""Evaluate a scalar reflected half-space Green function.

    For source and target in the upper half-space, the reflected field is

    .. math::

       G^R(\rho,h)=\frac{i}{4\pi}\int_0^\infty
       \frac{k_\rho}{k_{z1}} r(k_\rho) J_0(k_\rho\rho)
       e^{i k_{z1}h}\,dk_\rho,

    where ``h = z_source + z_target`` and
    ``r = (k_z1-k_z2)/(k_z1+k_z2)``.  Both media use the same scalar flux
    convention.  The fixed quadrature is preparation state; calls reuse the
    spectral weights and only rebuild target-dependent Bessel/phase factors.

    ``geometry_jet`` stores normalized derivatives with respect to ``rho``
    and ``h``.  The Bessel derivatives use the neighboring-order recurrence
    after all required orders have been evaluated.
    """

    def __init__(
        self,
        radial_wavenumbers: Any,
        quadrature_weights: Any,
        upper_wavenumber: float,
        lower_wavenumber: float,
        damping: float,
        *,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
    ) -> None:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        self.torch = torch
        self.device = torch.device(device)
        if complex_dtype in {"complex64", np.complex64, torch.complex64}:
            self.complex_dtype = torch.complex64
            self.real_dtype = torch.float32
        elif complex_dtype in {"complex128", np.complex128, torch.complex128}:
            self.complex_dtype = torch.complex128
            self.real_dtype = torch.float64
        else:
            raise ValueError("complex_dtype must be complex64 or complex128")

        radial = np.asarray(radial_wavenumbers, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        if (
            radial.ndim != 1
            or radial.size == 0
            or weights.shape != radial.shape
            or not np.all(np.isfinite(radial))
            or not np.all(np.isfinite(weights))
            or np.any(radial < 0.0)
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "radial_wavenumbers and quadrature_weights must be finite "
                "one-dimensional arrays with non-negative nodes and positive weights"
            )
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

        self.upper_wavenumber = upper_wavenumber
        self.lower_wavenumber = lower_wavenumber
        self.damping = damping
        self.radial_wavenumbers = torch.as_tensor(
            np.ascontiguousarray(radial), dtype=self.real_dtype, device=self.device
        )
        self.quadrature_weights = torch.as_tensor(
            np.ascontiguousarray(weights), dtype=self.real_dtype, device=self.device
        )
        upper_complex = torch.as_tensor(
            complex(upper_wavenumber, damping),
            dtype=self.complex_dtype,
            device=self.device,
        )
        lower_complex = torch.as_tensor(
            complex(lower_wavenumber, damping),
            dtype=self.complex_dtype,
            device=self.device,
        )
        radial_complex = self.radial_wavenumbers.to(dtype=self.complex_dtype)
        self.axial_upper = self._outgoing_root(upper_complex**2 - radial_complex**2)
        self.axial_lower = self._outgoing_root(lower_complex**2 - radial_complex**2)
        reflection = (self.axial_upper - self.axial_lower) / (
            self.axial_upper + self.axial_lower
        )
        self.reflection = reflection
        self.spectral_weights = (
            1j
            / (4.0 * math.pi)
            * self.quadrature_weights.to(dtype=self.complex_dtype)
            * radial_complex
            / self.axial_upper
            * reflection
        )
        self.prepared_mib = float(
            sum(
                tensor.nelement() * tensor.element_size()
                for tensor in (
                    self.radial_wavenumbers,
                    self.quadrature_weights,
                    self.axial_upper,
                    self.axial_lower,
                    self.reflection,
                    self.spectral_weights,
                )
            )
            / (1024.0 * 1024.0)
        )

    def _outgoing_root(self, value: Any) -> Any:
        root = self.torch.sqrt(value)
        flip = (root.imag < 0.0) | ((root.imag == 0.0) & (root.real < 0.0))
        return self.torch.where(flip, -root, root)

    def _geometry(self, rho: Any, height: Any) -> tuple[Any, Any, tuple[int, ...]]:
        radial = self.torch.as_tensor(rho, dtype=self.real_dtype, device=self.device)
        axial = self.torch.as_tensor(height, dtype=self.real_dtype, device=self.device)
        radial, axial = self.torch.broadcast_tensors(radial, axial)
        if not bool(
            self.torch.all(self.torch.isfinite(radial)).item()
            and self.torch.all(self.torch.isfinite(axial)).item()
        ):
            raise ValueError("rho and height must be finite")
        if bool(self.torch.any(radial < 0.0).item()) or bool(
            self.torch.any(axial <= 0.0).item()
        ):
            raise ValueError("rho must be non-negative and height must be positive")
        shape = tuple(radial.shape)
        return radial.reshape(-1), axial.reshape(-1), shape

    def _bessel_orders(self, rho: Any, max_order: int) -> Any:
        argument = (
            rho.detach().cpu().numpy()[:, None]
            * self.radial_wavenumbers.detach().cpu().numpy()[None, :]
        )
        orders = np.arange(-max_order, max_order + 1, dtype=np.int64)
        table = special.jv(orders[:, None, None], argument[None, :, :])
        return self.torch.as_tensor(
            np.ascontiguousarray(table),
            dtype=self.real_dtype,
            device=self.device,
        )

    def geometry_jet(
        self, rho: Any, height: Any, max_total_order: int
    ) -> dict[tuple[int, int], Any]:
        """Return normalized mixed derivatives ``(d_rho, d_height)``."""

        max_total_order = int(max_total_order)
        if max_total_order < 0:
            raise ValueError("max_total_order must be non-negative")
        radial, axial, shape = self._geometry(rho, height)
        bessel = self._bessel_orders(radial, max_total_order)
        q = self.radial_wavenumbers[None, :]
        phase = self.torch.exp(1j * axial[:, None] * self.axial_upper[None, :])
        common = phase * self.spectral_weights[None, :]
        output: dict[tuple[int, int], Any] = {}
        offset = max_total_order
        for a in range(max_total_order + 1):
            radial_derivative = self.torch.zeros_like(bessel[0])
            for j in range(a + 1):
                order = -a + 2 * j
                radial_derivative.add_(
                    ((-1) ** j) * math.comb(a, j) * bessel[offset + order]
                )
            radial_derivative.mul_(
                q**a / (2.0**a * math.factorial(a))
            )
            for b in range(max_total_order - a + 1):
                axial_factor = (
                    (1j * self.axial_upper[None, :]) ** b / math.factorial(b)
                )
                value = self.torch.sum(
                    common * radial_derivative * axial_factor, dim=-1
                )
                output[(a, b)] = value.reshape(shape)
        return output

    def evaluate(self, rho: Any, height: Any) -> Any:
        return self.geometry_jet(rho, height, 0)[(0, 0)]

    def evaluate_jet(
        self,
        jet: dict[tuple[int, int], Any],
        delta_rho: Any = 0.0,
        delta_height: Any = 0.0,
        *,
        max_total_order: int | None = None,
    ) -> Any:
        if not jet or (0, 0) not in jet:
            raise ValueError("jet must contain its zeroth-order coefficient")
        available_order = max(sum(key) for key in jet)
        order = available_order if max_total_order is None else int(max_total_order)
        if order < 0 or order > available_order:
            raise ValueError("max_total_order is outside the available jet")
        delta_radial = self.torch.as_tensor(
            delta_rho, dtype=self.real_dtype, device=self.device
        )
        delta_axial = self.torch.as_tensor(
            delta_height, dtype=self.real_dtype, device=self.device
        )
        output = self.torch.zeros_like(jet[(0, 0)])
        for (a, b), coefficient in jet.items():
            if a + b <= order:
                output = output + coefficient * delta_radial**a * delta_axial**b
        return output

    def geometry_jvp(
        self,
        rho: Any,
        height: Any,
        delta_rho: Any,
        delta_height: Any,
    ) -> Any:
        jet = self.geometry_jet(rho, height, 1)
        return jet[(1, 0)] * self.torch.as_tensor(
            delta_rho, dtype=self.real_dtype, device=self.device
        ) + jet[(0, 1)] * self.torch.as_tensor(
            delta_height, dtype=self.real_dtype, device=self.device
        )

    def geometry_vjp(
        self, rho: Any, height: Any, cotangent: Any
    ) -> tuple[Any, Any]:
        """Return the real-parameter VJP under the real complex inner product."""

        jet = self.geometry_jet(rho, height, 1)
        cotangent_tensor = self.torch.as_tensor(
            cotangent, dtype=self.complex_dtype, device=self.device
        )
        if tuple(cotangent_tensor.shape) != tuple(jet[(0, 0)].shape):
            raise ValueError("cotangent must match the broadcast geometry shape")
        radial = self.torch.real(self.torch.conj(cotangent_tensor) * jet[(1, 0)])
        axial = self.torch.real(self.torch.conj(cotangent_tensor) * jet[(0, 1)])
        return radial, axial

    def direct_green(self, rho: Any, axial_separation: Any) -> Any:
        radial = self.torch.as_tensor(rho, dtype=self.real_dtype, device=self.device)
        axial = self.torch.as_tensor(
            axial_separation, dtype=self.real_dtype, device=self.device
        )
        radial, axial = self.torch.broadcast_tensors(radial, axial)
        distance = self.torch.sqrt(radial**2 + axial**2)
        if bool(self.torch.any(distance <= 0.0).item()):
            raise ValueError("source and target must not coincide")
        complex_k = self.torch.as_tensor(
            complex(self.upper_wavenumber, self.damping),
            dtype=self.complex_dtype,
            device=self.device,
        )
        return self.torch.exp(1j * complex_k * distance) / (4.0 * math.pi * distance)

    def total_green(
        self, rho: Any, source_height: Any, target_height: Any
    ) -> Any:
        source = self.torch.as_tensor(
            source_height, dtype=self.real_dtype, device=self.device
        )
        target = self.torch.as_tensor(
            target_height, dtype=self.real_dtype, device=self.device
        )
        return self.direct_green(rho, target - source) + self.evaluate(
            rho, target + source
        )
