"""Differentiable exact-coordinate ACFO for selected azimuthal targets.

This operator preserves arbitrary source coordinates.  It evaluates one target
azimuth for every meridional q sample and exposes analytic first-order VJPs for
source weights, per-element form factors, q_perp, q_z and target azimuth.  The
integer harmonic support is frozen inside a prespecified q_perp envelope.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_finite_numpy(values: Any, *, name: str, ndim: int) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return np.ascontiguousarray(array)


class TorchDifferentiableExactCoordinateHarmonicOperator:
    """Selected-target exact-coordinate harmonic forward and analytic VJP."""

    def __init__(
        self,
        coordinates: Any,
        element_indices: Any,
        q_perp: Any,
        q_z: Any,
        phi: Any,
        *,
        n_elements: int,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
        harmonic_padding: int = 48,
        miller_margin: int = 32,
        q_block_size: int = 8,
        support_q_perp: Any | None = None,
    ) -> None:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        self.torch = torch
        self.device = torch.device(device)
        if complex_dtype in {"complex64", np.complex64, torch.complex64}:
            self.complex_dtype = torch.complex64
            self._numpy_complex_dtype = np.dtype(np.complex64)
        elif complex_dtype in {"complex128", np.complex128, torch.complex128}:
            self.complex_dtype = torch.complex128
            self._numpy_complex_dtype = np.dtype(np.complex128)
        else:
            raise ValueError("complex_dtype must be complex64 or complex128")
        self.real_dtype = torch.float64

        coords = _as_finite_numpy(coordinates, name="coordinates", ndim=2).astype(
            np.float64, copy=False
        )
        if coords.shape[1] != 3 or coords.shape[0] == 0:
            raise ValueError("coordinates must have shape (n_source, 3)")
        elements = np.asarray(element_indices, dtype=np.int64)
        if elements.shape != (coords.shape[0],):
            raise ValueError("element_indices must have shape (n_source,)")
        n_elements = int(n_elements)
        if n_elements <= 0 or np.any(elements < 0) or np.any(elements >= n_elements):
            raise ValueError("element_indices contains an out-of-range value")
        qp = _as_finite_numpy(q_perp, name="q_perp", ndim=1).astype(
            np.float64, copy=False
        )
        qz = _as_finite_numpy(q_z, name="q_z", ndim=1).astype(np.float64, copy=False)
        azimuth = _as_finite_numpy(phi, name="phi", ndim=1).astype(
            np.float64, copy=False
        )
        if qp.size == 0 or qz.shape != qp.shape or azimuth.shape != qp.shape:
            raise ValueError("q_perp, q_z and phi must have equal non-empty shape")
        if np.any(qp < 0.0):
            raise ValueError("q_perp must be non-negative")
        if support_q_perp is None:
            support = qp.copy()
        else:
            support = _as_finite_numpy(
                support_q_perp, name="support_q_perp", ndim=1
            ).astype(np.float64, copy=False)
            if support.size == 1:
                support = np.full_like(qp, support.item())
            if support.shape != qp.shape or np.any(support < qp):
                raise ValueError("support_q_perp must bound q_perp and match its shape")

        harmonic_padding = int(harmonic_padding)
        miller_margin = int(miller_margin)
        q_block_size = int(q_block_size)
        if harmonic_padding < 0 or miller_margin < 0 or q_block_size <= 0:
            raise ValueError("padding, Miller margin and q block size are invalid")
        radius = np.hypot(coords[:, 0], coords[:, 1])
        beta = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
        cutoffs = np.ceil(support * float(radius.max(initial=0.0))).astype(np.int64)
        cutoffs += harmonic_padding
        max_cutoff = int(cutoffs.max(initial=0))
        derivative_max_order = max_cutoff + 1
        virtual_n_phi = 1
        while virtual_n_phi <= 2 * derivative_max_order + 2:
            virtual_n_phi *= 2

        self.n_source = int(coords.shape[0])
        self.n_q = int(qp.size)
        self.n_elements = n_elements
        self.harmonic_padding = harmonic_padding
        self.miller_margin = miller_margin
        self.q_block_size = min(q_block_size, self.n_q)
        self.max_cutoff = max_cutoff
        self.derivative_max_order = derivative_max_order
        self.virtual_n_phi = virtual_n_phi
        self.radius = torch.as_tensor(radius, dtype=self.real_dtype, device=self.device)
        self.beta = torch.as_tensor(beta, dtype=self.real_dtype, device=self.device)
        self.z = torch.as_tensor(coords[:, 2], dtype=self.real_dtype, device=self.device)
        self.element_indices = torch.as_tensor(
            elements, dtype=torch.long, device=self.device
        )
        self.default_q_perp = torch.as_tensor(qp, dtype=self.real_dtype, device=self.device)
        self.default_q_z = torch.as_tensor(qz, dtype=self.real_dtype, device=self.device)
        self.default_phi = torch.as_tensor(
            azimuth, dtype=self.real_dtype, device=self.device
        )
        self.support_q_perp = torch.as_tensor(
            support, dtype=self.real_dtype, device=self.device
        )
        self.support_cutoffs = torch.as_tensor(
            cutoffs, dtype=torch.long, device=self.device
        )
        signed_modes = np.arange(-max_cutoff, max_cutoff + 1, dtype=np.int64)
        self.signed_modes = torch.as_tensor(
            signed_modes, dtype=torch.long, device=self.device
        )
        self.mode_abs = torch.abs(self.signed_modes)
        self.left_neighbor_abs = torch.abs(self.signed_modes - 1)
        self.right_neighbor_abs = torch.abs(self.signed_modes + 1)

    def _resolve_geometry(
        self, q_perp: Any | None, q_z: Any | None, phi: Any | None
    ) -> tuple[Any, Any, Any]:
        torch = self.torch
        qp = self.default_q_perp if q_perp is None else torch.as_tensor(
            q_perp, dtype=self.real_dtype, device=self.device
        )
        qz = self.default_q_z if q_z is None else torch.as_tensor(
            q_z, dtype=self.real_dtype, device=self.device
        )
        azimuth = self.default_phi if phi is None else torch.as_tensor(
            phi, dtype=self.real_dtype, device=self.device
        )
        if tuple(qp.shape) != (self.n_q,) or tuple(qz.shape) != (self.n_q,) or tuple(
            azimuth.shape
        ) != (self.n_q,):
            raise ValueError(f"geometry arrays must have shape ({self.n_q},)")
        if not bool(torch.all(torch.isfinite(qp)).item()) or not bool(
            torch.all(torch.isfinite(qz)).item()
        ) or not bool(torch.all(torch.isfinite(azimuth)).item()):
            raise ValueError("geometry arrays must be finite")
        if bool(torch.any(qp < 0.0).item()):
            raise ValueError("q_perp must be non-negative")
        required = torch.ceil(qp.detach() * torch.max(self.radius)).to(torch.long)
        required += self.harmonic_padding
        if bool(torch.any(required > self.support_cutoffs).item()):
            raise ValueError("q_perp exceeds the frozen support envelope")
        return qp, qz, azimuth

    def _weights(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != (self.n_source,):
            raise ValueError(f"source weights must have shape ({self.n_source},)")
        return tensor

    def _weight_batch(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tensor.ndim != 2 or tensor.shape[1] != self.n_source:
            raise ValueError(
                f"batched source weights must have shape (n_batch, {self.n_source})"
            )
        return tensor

    def _form_factors(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != (self.n_elements, self.n_q):
            raise ValueError(
                f"form factors must have shape ({self.n_elements}, {self.n_q})"
            )
        return tensor

    def _data(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != (self.n_q,):
            raise ValueError(f"data must have shape ({self.n_q},)")
        return tensor

    def _data_batch(self, values: Any, n_batch: int) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != (n_batch, self.n_q):
            raise ValueError(
                f"batched data must have shape ({n_batch}, {self.n_q})"
            )
        return tensor

    def _positive_kernel(self, q_perp: Any, max_order: int) -> Any:
        if self.device.type == "cuda":
            from .gpu_miller import (
                gpu_miller_kernel128_torch_resident,
                gpu_miller_kernel64_torch_resident,
            )

            builder = (
                gpu_miller_kernel64_torch_resident
                if self.complex_dtype == self.torch.complex64
                else gpu_miller_kernel128_torch_resident
            )
            table = builder(
                q_perp.detach(),
                self.radius,
                n_phi=self.virtual_n_phi,
                max_cutoff=max_order,
                extra_order=self.miller_margin,
                torch=self.torch,
            )
            return table / float(self.virtual_n_phi)
        from scipy import special

        qp = q_perp.detach().cpu().numpy()
        radius = self.radius.detach().cpu().numpy()
        orders = np.arange(max_order + 1)
        table = special.jv(
            orders[None, None, :], qp[:, None, None] * radius[None, :, None]
        ) * np.power(1j, orders)[None, None, :]
        return self.torch.as_tensor(
            np.ascontiguousarray(table, dtype=self._numpy_complex_dtype),
            dtype=self.complex_dtype,
            device=self.device,
        )

    def _block_terms(
        self,
        q_perp: Any,
        q_z: Any,
        phi: Any,
        start: int,
        stop: int,
        *,
        derivatives: bool,
    ) -> tuple[Any, Any | None, Any | None, Any | None]:
        torch = self.torch
        maximum_order = self.derivative_max_order if derivatives else self.max_cutoff
        positive = self._positive_kernel(q_perp[start:stop], maximum_order)
        kernel = positive.index_select(2, self.mode_abs)
        mask = self.mode_abs[None, :] <= self.support_cutoffs[start:stop, None]
        kernel = kernel * mask[:, None, :]
        angle = phi[start:stop, None] - self.beta[None, :]
        phase = torch.exp(
            1j
            * angle[:, :, None]
            * self.signed_modes.to(dtype=self.real_dtype)[None, None, :]
        ).to(dtype=self.complex_dtype)
        axial = torch.exp(1j * q_z[start:stop, None] * self.z[None, :]).to(
            dtype=self.complex_dtype
        )
        harmonic = torch.sum(kernel * phase, dim=-1)
        base = axial * harmonic
        if not derivatives:
            return base, None, None, None
        left = positive.index_select(2, self.left_neighbor_abs) * mask[:, None, :]
        right = positive.index_select(2, self.right_neighbor_abs) * mask[:, None, :]
        dkernel = 0.5j * self.radius[None, :, None] * (left + right)
        d_perp = axial * torch.sum(dkernel * phase, dim=-1)
        d_z = 1j * self.z[None, :] * base
        d_phi = axial * torch.sum(
            kernel
            * phase
            * (1j * self.signed_modes.to(dtype=self.real_dtype))[None, None, :],
            dim=-1,
        )
        return base, d_perp, d_z, d_phi

    def _forward_resolved(
        self, weights: Any, form_factors: Any, q_perp: Any, q_z: Any, phi: Any
    ) -> Any:
        output = self.torch.empty(
            self.n_q, dtype=self.complex_dtype, device=self.device
        )
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            base, _, _, _ = self._block_terms(
                q_perp, q_z, phi, start, stop, derivatives=False
            )
            factors = form_factors[:, start:stop].transpose(0, 1)[
                :, self.element_indices
            ]
            output[start:stop] = self.torch.sum(
                weights[None, :] * factors * base, dim=1
            )
        return output

    def forward(
        self,
        source_weights: Any,
        form_factors: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> Any:
        weights = self._weights(source_weights)
        factors = self._form_factors(form_factors)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        return self._forward_resolved(weights, factors, qp, qz, azimuth)

    def _forward_batch_resolved(
        self, weights: Any, form_factors: Any, q_perp: Any, q_z: Any, phi: Any
    ) -> Any:
        output = self.torch.empty(
            (weights.shape[0], self.n_q),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            base, _, _, _ = self._block_terms(
                q_perp, q_z, phi, start, stop, derivatives=False
            )
            factors = form_factors[:, start:stop].transpose(0, 1)[
                :, self.element_indices
            ]
            output[:, start:stop] = weights @ (factors * base).transpose(0, 1)
        return output

    def forward_batch(
        self,
        source_weights: Any,
        form_factors: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> Any:
        """Evaluate a batch of source states while sharing the geometry kernel."""
        weights = self._weight_batch(source_weights)
        factors = self._form_factors(form_factors)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        return self._forward_batch_resolved(weights, factors, qp, qz, azimuth)

    def _vjp(
        self,
        weights: Any,
        form_factors: Any,
        q_perp: Any,
        q_z: Any,
        phi: Any,
        cotangent: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        torch = self.torch
        weight_gradient = torch.zeros_like(weights)
        factor_gradient = torch.zeros_like(form_factors)
        gradient_perp = torch.empty(
            self.n_q, dtype=self.real_dtype, device=self.device
        )
        gradient_z = torch.empty_like(gradient_perp)
        gradient_phi = torch.empty_like(gradient_perp)
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            base, d_perp, d_z, d_phi = self._block_terms(
                q_perp, q_z, phi, start, stop, derivatives=True
            )
            factors = form_factors[:, start:stop].transpose(0, 1)[
                :, self.element_indices
            ]
            local_cotangent = cotangent[start:stop]
            weight_gradient.add_(
                torch.sum(
                    torch.conj(factors * base) * local_cotangent[:, None], dim=0
                )
            )
            source_factor_gradient = (
                torch.conj(weights[None, :] * base) * local_cotangent[:, None]
            )
            for element in range(self.n_elements):
                selected = self.element_indices == element
                factor_gradient[element, start:stop] = torch.sum(
                    source_factor_gradient[:, selected], dim=1
                )
            weighted = weights[None, :] * factors
            for destination, derivative in (
                (gradient_perp, d_perp),
                (gradient_z, d_z),
                (gradient_phi, d_phi),
            ):
                derivative_amplitude = torch.sum(weighted * derivative, dim=1)
                destination[start:stop] = torch.real(
                    torch.conj(local_cotangent) * derivative_amplitude
                )
        return (
            weight_gradient,
            factor_gradient,
            gradient_perp,
            gradient_z,
            gradient_phi,
        )

    def _vjp_batch(
        self,
        weights: Any,
        form_factors: Any,
        q_perp: Any,
        q_z: Any,
        phi: Any,
        cotangent: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        torch = self.torch
        weight_gradient = torch.zeros_like(weights)
        factor_gradient = torch.zeros_like(form_factors)
        gradient_perp = torch.empty(
            self.n_q, dtype=self.real_dtype, device=self.device
        )
        gradient_z = torch.empty_like(gradient_perp)
        gradient_phi = torch.empty_like(gradient_perp)
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            base, d_perp, d_z, d_phi = self._block_terms(
                q_perp, q_z, phi, start, stop, derivatives=True
            )
            factors = form_factors[:, start:stop].transpose(0, 1)[
                :, self.element_indices
            ]
            coefficient = factors * base
            local_cotangent = cotangent[:, start:stop]
            weight_gradient.add_(local_cotangent @ torch.conj(coefficient))
            for element in range(self.n_elements):
                selected = self.element_indices == element
                factor_gradient[element, start:stop] = torch.einsum(
                    "bq,bs,qs->q",
                    local_cotangent,
                    torch.conj(weights[:, selected]),
                    torch.conj(base[:, selected]),
                )
            selected_factors = factors
            for destination, derivative in (
                (gradient_perp, d_perp),
                (gradient_z, d_z),
                (gradient_phi, d_phi),
            ):
                derivative_amplitude = weights @ (
                    selected_factors * derivative
                ).transpose(0, 1)
                destination[start:stop] = torch.real(
                    torch.sum(torch.conj(local_cotangent) * derivative_amplitude, dim=0)
                )
        return (
            weight_gradient,
            factor_gradient,
            gradient_perp,
            gradient_z,
            gradient_phi,
        )

    def adjoint_weights(
        self,
        data_values: Any,
        form_factors: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> Any:
        data = self._data(data_values)
        factors = self._form_factors(form_factors)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        zero_weights = self.torch.zeros(
            self.n_source, dtype=self.complex_dtype, device=self.device
        )
        return self._vjp(zero_weights, factors, qp, qz, azimuth, data)[0]

    def geometry_vjp(
        self,
        source_weights: Any,
        form_factors: Any,
        data_cotangent: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        weights = self._weights(source_weights)
        factors = self._form_factors(form_factors)
        data = self._data(data_cotangent)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        result = self._vjp(weights, factors, qp, qz, azimuth, data)
        return result[2], result[3], result[4]

    def autograd_forward(
        self,
        source_weights: Any,
        form_factors: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> Any:
        weights = self._weights(source_weights)
        factors = self._form_factors(form_factors)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        plan = self
        torch = self.torch

        class _ExactCoordinateAcfoFunction(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                source_tensor: Any,
                factor_tensor: Any,
                q_perp_tensor: Any,
                q_z_tensor: Any,
                phi_tensor: Any,
            ) -> Any:
                ctx.save_for_backward(
                    source_tensor,
                    factor_tensor,
                    q_perp_tensor,
                    q_z_tensor,
                    phi_tensor,
                )
                return plan._forward_resolved(
                    source_tensor,
                    factor_tensor,
                    q_perp_tensor,
                    q_z_tensor,
                    phi_tensor,
                )

            @staticmethod
            def backward(ctx: Any, data_cotangent: Any) -> tuple[Any, Any, Any, Any, Any]:
                source_tensor, factor_tensor, qp_tensor, qz_tensor, phi_tensor = (
                    ctx.saved_tensors
                )
                values = plan._vjp(
                    source_tensor,
                    factor_tensor,
                    qp_tensor,
                    qz_tensor,
                    phi_tensor,
                    data_cotangent.to(dtype=plan.complex_dtype),
                )
                return tuple(
                    value if ctx.needs_input_grad[index] else None
                    for index, value in enumerate(values)
                )

        return _ExactCoordinateAcfoFunction.apply(weights, factors, qp, qz, azimuth)

    def autograd_forward_batch(
        self,
        source_weights: Any,
        form_factors: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        phi: Any | None = None,
    ) -> Any:
        """Differentiable batched forward with one shared geometry/form-factor state."""
        weights = self._weight_batch(source_weights)
        factors = self._form_factors(form_factors)
        qp, qz, azimuth = self._resolve_geometry(q_perp, q_z, phi)
        plan = self
        torch = self.torch

        class _ExactCoordinateAcfoBatchFunction(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                source_tensor: Any,
                factor_tensor: Any,
                q_perp_tensor: Any,
                q_z_tensor: Any,
                phi_tensor: Any,
            ) -> Any:
                ctx.save_for_backward(
                    source_tensor,
                    factor_tensor,
                    q_perp_tensor,
                    q_z_tensor,
                    phi_tensor,
                )
                return plan._forward_batch_resolved(
                    source_tensor,
                    factor_tensor,
                    q_perp_tensor,
                    q_z_tensor,
                    phi_tensor,
                )

            @staticmethod
            def backward(ctx: Any, data_cotangent: Any) -> tuple[Any, Any, Any, Any, Any]:
                source_tensor, factor_tensor, qp_tensor, qz_tensor, phi_tensor = (
                    ctx.saved_tensors
                )
                values = plan._vjp_batch(
                    source_tensor,
                    factor_tensor,
                    qp_tensor,
                    qz_tensor,
                    phi_tensor,
                    data_cotangent.to(dtype=plan.complex_dtype),
                )
                return tuple(
                    value if ctx.needs_input_grad[index] else None
                    for index, value in enumerate(values)
                )

        return _ExactCoordinateAcfoBatchFunction.apply(
            weights, factors, qp, qz, azimuth
        )

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
