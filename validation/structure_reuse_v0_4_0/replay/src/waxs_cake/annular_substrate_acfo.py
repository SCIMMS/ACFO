"""Modal reflected-Green contractions for annular Born sources above a substrate."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import special

from .sommerfeld_acfo import ScalarPlanarHalfspaceSommerfeldOperator


def _finite_positive_vector(values: Any, *, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError(f"{name} must be a non-empty finite positive vector")
    return np.ascontiguousarray(array)


class ScalarSubstrateAnnularBornOperator:
    r"""Apply a reflected half-space Green function between complete rings.

    Graf's addition theorem gives the angular Fourier coefficients

    .. math::

       C_m(r_t,r_s,h)=\frac{i}{4\pi}\int_0^\infty
       \frac{k_\rho}{k_{z1}}r(k_\rho)
       J_m(k_\rho r_t)J_m(k_\rho r_s)e^{ik_{z1}h}\,dk_\rho.

    The forward map is the angular convolution of this kernel with a sampled
    Born source on each source ring.  The radial source quadrature weights are
    frozen preparation state.  Geometry derivatives therefore differentiate
    the Green kernel only, not a moving source quadrature rule.
    """

    def __init__(
        self,
        halfspace: ScalarPlanarHalfspaceSommerfeldOperator,
        source_radii: Any,
        target_radii: Any,
        source_heights: Any,
        target_heights: Any,
        source_weights: Any,
        n_phi: int,
        *,
        support_q_max: float,
        harmonic_padding: int = 48,
        miller_margin: int = 32,
    ) -> None:
        self.halfspace = halfspace
        self.torch = halfspace.torch
        self.device = halfspace.device
        self.real_dtype = halfspace.real_dtype
        self.complex_dtype = halfspace.complex_dtype
        source_radii_np = _finite_positive_vector(source_radii, name="source_radii")
        target_radii_np = _finite_positive_vector(target_radii, name="target_radii")
        source_heights_np = _finite_positive_vector(
            source_heights, name="source_heights"
        )
        target_heights_np = _finite_positive_vector(
            target_heights, name="target_heights"
        )
        source_weights_np = _finite_positive_vector(
            source_weights, name="source_weights"
        )
        if source_heights_np.shape != source_radii_np.shape:
            raise ValueError("source_heights must match source_radii")
        if target_heights_np.shape != target_radii_np.shape:
            raise ValueError("target_heights must match target_radii")
        if source_weights_np.shape != source_radii_np.shape:
            raise ValueError("source_weights must match source_radii")
        n_phi = int(n_phi)
        harmonic_padding = int(harmonic_padding)
        miller_margin = int(miller_margin)
        support_q_max = float(support_q_max)
        if n_phi < 4 or harmonic_padding < 0 or miller_margin < 0:
            raise ValueError("invalid angular support parameters")
        if not np.isfinite(support_q_max) or support_q_max <= 0.0:
            raise ValueError("support_q_max must be finite and positive")
        sampled_q_max = float(self.torch.max(halfspace.radial_wavenumbers).cpu())
        if support_q_max < sampled_q_max:
            raise ValueError("support_q_max must bound all quadrature nodes")
        maximum_radius = max(source_radii_np.max(), target_radii_np.max())
        max_harmonic = int(math.ceil(support_q_max * maximum_radius)) + harmonic_padding
        if max_harmonic >= n_phi // 2:
            raise ValueError(
                "qR+padding reaches angular Nyquist; increase n_phi so H < n_phi/2"
            )

        self.n_phi = n_phi
        self.n_source_rings = int(source_radii_np.size)
        self.n_target_rings = int(target_radii_np.size)
        self.object_shape = (self.n_source_rings, n_phi)
        self.data_shape = (self.n_target_rings, n_phi)
        self.support_q_max = support_q_max
        self.harmonic_padding = harmonic_padding
        self.miller_margin = miller_margin
        self.max_harmonic = max_harmonic
        self.source_radii = self.torch.as_tensor(
            source_radii_np, dtype=self.real_dtype, device=self.device
        )
        self.target_radii = self.torch.as_tensor(
            target_radii_np, dtype=self.real_dtype, device=self.device
        )
        self.source_heights = self.torch.as_tensor(
            source_heights_np, dtype=self.real_dtype, device=self.device
        )
        self.target_heights = self.torch.as_tensor(
            target_heights_np, dtype=self.real_dtype, device=self.device
        )
        self.source_weights = self.torch.as_tensor(
            source_weights_np, dtype=self.real_dtype, device=self.device
        )
        signed_modes_np = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
        mode_abs_np = np.abs(signed_modes_np)
        support_mask_np = mode_abs_np <= max_harmonic
        self.signed_modes = self.torch.as_tensor(
            signed_modes_np, dtype=self.torch.long, device=self.device
        )
        self.mode_abs = self.torch.as_tensor(
            mode_abs_np, dtype=self.torch.long, device=self.device
        )
        self.support_mask = self.torch.as_tensor(
            support_mask_np, dtype=self.torch.bool, device=self.device
        )

        kernels = self._prepare_modal_kernels(
            source_radii_np,
            target_radii_np,
            source_heights_np,
            target_heights_np,
            mode_abs_np,
            support_mask_np,
        )
        (
            self.modal_kernel,
            self.modal_derivative_target_radius,
            self.modal_derivative_source_radius,
            self.modal_derivative_height,
            self.modal_derivative_substrate_thickness,
        ) = kernels
        resident = (
            self.source_radii,
            self.target_radii,
            self.source_heights,
            self.target_heights,
            self.source_weights,
            self.signed_modes,
            self.mode_abs,
            self.support_mask,
            *(kernel for kernel in kernels if kernel is not None),
        )
        self.prepared_mib = float(
            sum(tensor.nelement() * tensor.element_size() for tensor in resident)
            / (1024.0 * 1024.0)
        )

    def _prepare_modal_kernels(
        self,
        source_radii: np.ndarray,
        target_radii: np.ndarray,
        source_heights: np.ndarray,
        target_heights: np.ndarray,
        mode_abs: np.ndarray,
        support_mask: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any | None]:
        q_np = np.asarray(
            self.halfspace.radial_wavenumbers.detach().cpu(), dtype=np.float64
        )
        orders = np.arange(self.max_harmonic + 2, dtype=np.int64)
        source_bessel_np = special.jv(
            orders[:, None, None],
            q_np[None, :, None] * source_radii[None, None, :],
        )
        target_bessel_np = special.jv(
            orders[:, None, None],
            q_np[None, :, None] * target_radii[None, None, :],
        )

        def derivative(table: np.ndarray) -> np.ndarray:
            output = np.empty(
                (self.max_harmonic + 1, table.shape[1], table.shape[2]),
                dtype=np.float64,
            )
            output[0] = -q_np[:, None] * table[1]
            if self.max_harmonic:
                output[1:] = 0.5 * q_np[None, :, None] * (
                    table[: self.max_harmonic] - table[2 : self.max_harmonic + 2]
                )
            return output

        derivative_source_np = derivative(source_bessel_np)
        derivative_target_np = derivative(target_bessel_np)
        source_bessel = self.torch.as_tensor(
            np.ascontiguousarray(source_bessel_np[: self.max_harmonic + 1]),
            dtype=self.real_dtype,
            device=self.device,
        )
        target_bessel = self.torch.as_tensor(
            np.ascontiguousarray(target_bessel_np[: self.max_harmonic + 1]),
            dtype=self.real_dtype,
            device=self.device,
        )
        derivative_source = self.torch.as_tensor(
            np.ascontiguousarray(derivative_source_np),
            dtype=self.real_dtype,
            device=self.device,
        )
        derivative_target = self.torch.as_tensor(
            np.ascontiguousarray(derivative_target_np),
            dtype=self.real_dtype,
            device=self.device,
        )
        heights = self.target_heights[:, None] + self.source_heights[None, :]
        phase = self.torch.exp(
            1j
            * heights[:, :, None]
            * self.halfspace.axial_upper[None, None, :]
        )
        spectral = self.halfspace.spectral_weights
        axial_spectral = spectral * (1j * self.halfspace.axial_upper)

        def contract(
            target_table: Any, source_table: Any, spectral_table: Any
        ) -> Any:
            positive = self.torch.einsum(
                "tsq,mqt,mqs,q->tsm",
                phase,
                target_table,
                source_table,
                spectral_table,
            )
            full = self.torch.zeros(
                (self.n_target_rings, self.n_source_rings, self.n_phi),
                dtype=self.complex_dtype,
                device=self.device,
            )
            supported_indices = np.flatnonzero(support_mask)
            positive_indices = mode_abs[supported_indices]
            full[:, :, supported_indices] = positive[:, :, positive_indices]
            return full

        thickness_spectral = getattr(
            self.halfspace, "thickness_derivative_spectral_weights", None
        )
        thickness_kernel = (
            None
            if thickness_spectral is None
            else contract(target_bessel, source_bessel, thickness_spectral)
        )
        return (
            contract(target_bessel, source_bessel, spectral),
            contract(derivative_target, source_bessel, spectral),
            contract(target_bessel, derivative_source, spectral),
            contract(target_bessel, source_bessel, axial_spectral),
            thickness_kernel,
        )

    def _object_tensor(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != self.object_shape or not bool(
            self.torch.all(self.torch.isfinite(tensor)).item()
        ):
            raise ValueError(f"object values must be finite with shape {self.object_shape}")
        return tensor

    def _data_tensor(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != self.data_shape or not bool(
            self.torch.all(self.torch.isfinite(tensor)).item()
        ):
            raise ValueError(f"data values must be finite with shape {self.data_shape}")
        return tensor

    def _apply_modal_kernel(self, values: Any, modal_kernel: Any) -> Any:
        weighted = values * self.source_weights[:, None]
        source_modes = self.torch.fft.fft(weighted, dim=-1)
        target_modes = 2.0 * math.pi * self.torch.einsum(
            "tsm,sm->tm", modal_kernel, source_modes
        )
        return self.torch.fft.ifft(target_modes, dim=-1)

    def forward(self, source_values: Any) -> Any:
        return self._apply_modal_kernel(
            self._object_tensor(source_values), self.modal_kernel
        )

    def adjoint(self, data_values: Any) -> Any:
        data = self._data_tensor(data_values)
        target_modes = self.torch.fft.fft(data, dim=-1)
        source_modes = 2.0 * math.pi * self.torch.einsum(
            "tsm,tm->sm", self.torch.conj(self.modal_kernel), target_modes
        )
        return (
            self.torch.fft.ifft(source_modes, dim=-1)
            * self.source_weights[:, None]
        )

    def geometry_jvp(
        self,
        source_values: Any,
        delta_target_radius: Any,
        delta_source_radius: Any,
        delta_target_height: Any,
        delta_source_height: Any,
    ) -> Any:
        values = self._object_tensor(source_values)
        delta_rt = self.torch.as_tensor(
            delta_target_radius, dtype=self.real_dtype, device=self.device
        )
        delta_rs = self.torch.as_tensor(
            delta_source_radius, dtype=self.real_dtype, device=self.device
        )
        delta_zt = self.torch.as_tensor(
            delta_target_height, dtype=self.real_dtype, device=self.device
        )
        delta_zs = self.torch.as_tensor(
            delta_source_height, dtype=self.real_dtype, device=self.device
        )
        if tuple(delta_rt.shape) != (self.n_target_rings,) or tuple(
            delta_zt.shape
        ) != (self.n_target_rings,):
            raise ValueError("target geometry directions have incorrect shape")
        if tuple(delta_rs.shape) != (self.n_source_rings,) or tuple(
            delta_zs.shape
        ) != (self.n_source_rings,):
            raise ValueError("source geometry directions have incorrect shape")
        delta_kernel = (
            self.modal_derivative_target_radius * delta_rt[:, None, None]
            + self.modal_derivative_source_radius * delta_rs[None, :, None]
            + self.modal_derivative_height
            * (delta_zt[:, None] + delta_zs[None, :])[:, :, None]
        )
        return self._apply_modal_kernel(values, delta_kernel)

    def geometry_vjp(
        self, source_values: Any, cotangent: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return fused real-parameter VJPs for ring radii and heights."""

        values = self._object_tensor(source_values)
        data = self._data_tensor(cotangent)
        source_modes = self.torch.fft.fft(
            values * self.source_weights[:, None], dim=-1
        )
        data_modes = self.torch.fft.fft(data, dim=-1)
        pair = (
            (2.0 * math.pi / self.n_phi)
            * self.torch.conj(data_modes)[:, None, :]
            * source_modes[None, :, :]
        )
        gradient_rt = self.torch.real(
            self.torch.sum(pair * self.modal_derivative_target_radius, dim=(1, 2))
        )
        gradient_rs = self.torch.real(
            self.torch.sum(pair * self.modal_derivative_source_radius, dim=(0, 2))
        )
        pair_height = self.torch.real(
            self.torch.sum(pair * self.modal_derivative_height, dim=-1)
        )
        gradient_zt = self.torch.sum(pair_height, dim=1)
        gradient_zs = self.torch.sum(pair_height, dim=0)
        return gradient_rt, gradient_rs, gradient_zt, gradient_zs

    def substrate_thickness_jvp(
        self, source_values: Any, delta_thickness: Any = 1.0
    ) -> Any:
        """Differentiate the annular action with respect to slab thickness."""

        if self.modal_derivative_substrate_thickness is None:
            raise ValueError("the substrate operator has no thickness derivative")
        delta = self.torch.as_tensor(
            delta_thickness, dtype=self.real_dtype, device=self.device
        )
        if delta.ndim != 0 or not bool(self.torch.isfinite(delta).item()):
            raise ValueError("delta_thickness must be a finite scalar")
        return delta * self._apply_modal_kernel(
            self._object_tensor(source_values),
            self.modal_derivative_substrate_thickness,
        )

    def substrate_thickness_vjp(self, source_values: Any, cotangent: Any) -> Any:
        """Return the real thickness VJP paired with a complex cotangent."""

        if self.modal_derivative_substrate_thickness is None:
            raise ValueError("the substrate operator has no thickness derivative")
        values = self._object_tensor(source_values)
        data = self._data_tensor(cotangent)
        source_modes = self.torch.fft.fft(
            values * self.source_weights[:, None], dim=-1
        )
        data_modes = self.torch.fft.fft(data, dim=-1)
        pair = (
            (2.0 * math.pi / self.n_phi)
            * self.torch.conj(data_modes)[:, None, :]
            * source_modes[None, :, :]
        )
        return self.torch.real(
            self.torch.sum(pair * self.modal_derivative_substrate_thickness)
        )
