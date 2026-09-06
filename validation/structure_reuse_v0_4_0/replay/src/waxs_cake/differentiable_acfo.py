"""Differentiable qR-adaptive ACFO operator.

The operator keeps a frozen, prespecified harmonic support and differentiates
the continuous Fourier kernel inside that support.  Radial derivatives use the
neighboring-order identity for ``K_m = n_phi * i**m * J_m(q_perp * r)``::

    d K_m / d q_perp = 0.5j * r * (K_{m-1} + K_{m+1}).

The two neighboring contractions are accumulated before the ``0.5j`` factor
is applied.  No shifted copy of the complete kernel and no derivative Bessel
table are retained.  Integer support selection itself is deliberately outside
the differentiable map.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _numpy_1d(values: Any, *, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return np.ascontiguousarray(array)


class TorchDifferentiableAxisymmetricOperator:
    """Compact ACFO forward/adjoint with analytic geometry JVP and VJP.

    Parameters
    ----------
    r_centers, z_centers
        Cylindrical source-grid coordinates.
    q_perp, q_z
        Default meridional target coordinates.  ``q_perp`` is a non-negative
        radial magnitude.
    n_phi
        Size of the complete uniform SO(2) orbit and the source angular grid.
    support_q_perp
        Optional per-target upper bound used to freeze harmonic support.  Any
        geometry passed later must remain inside this envelope.  Supplying an
        explicit envelope is recommended for geometry optimization.

    Notes
    -----
    The object shape is ``(n_r, n_z, n_phi)`` and the data shape is
    ``(n_q, n_phi)``.  Form-factor derivatives are not included; callers must
    fold only q-independent coefficients into the object values.
    """

    def __init__(
        self,
        r_centers: Any,
        z_centers: Any,
        q_perp: Any,
        q_z: Any,
        n_phi: int,
        *,
        torch: Any | None = None,
        device: Any = "cpu",
        complex_dtype: Any = "complex128",
        harmonic_padding: int = 48,
        miller_margin: int = 32,
        q_block_size: int = 64,
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

        r_np = _numpy_1d(r_centers, name="r_centers")
        z_np = _numpy_1d(z_centers, name="z_centers")
        q_perp_np = _numpy_1d(q_perp, name="q_perp")
        q_z_np = _numpy_1d(q_z, name="q_z")
        if q_z_np.shape != q_perp_np.shape:
            raise ValueError("q_perp and q_z must have equal shape")
        if np.any(q_perp_np < 0.0):
            raise ValueError("q_perp must be non-negative")
        n_phi = int(n_phi)
        if n_phi < 3:
            raise ValueError("n_phi must be at least 3")
        harmonic_padding = int(harmonic_padding)
        miller_margin = int(miller_margin)
        q_block_size = int(q_block_size)
        if harmonic_padding < 0 or miller_margin < 0:
            raise ValueError("harmonic_padding and miller_margin must be non-negative")
        if q_block_size <= 0:
            raise ValueError("q_block_size must be positive")

        if support_q_perp is None:
            support_np = q_perp_np.copy()
        else:
            support_np = np.asarray(
                _numpy_1d(support_q_perp, name="support_q_perp"),
                dtype=np.float64,
            )
            if support_np.size == 1:
                support_np = np.full(q_perp_np.shape, support_np.item())
            if support_np.shape != q_perp_np.shape:
                raise ValueError("support_q_perp must be scalar or match q_perp")
            if np.any(support_np < q_perp_np):
                raise ValueError("support_q_perp must bound the default q_perp")

        cutoffs_np = (
            np.ceil(support_np[:, None] * np.abs(r_np)[None, :]).astype(np.int64)
            + harmonic_padding
        )
        max_cutoff = int(cutoffs_np.max(initial=0))
        derivative_max_order = max_cutoff + 1
        if derivative_max_order >= n_phi // 2:
            raise ValueError(
                "derivative harmonic support reaches angular Nyquist; increase n_phi "
                "so H + 1 < n_phi / 2"
            )

        self.n_phi = n_phi
        self.n_q = int(q_perp_np.size)
        self.object_shape = (int(r_np.size), int(z_np.size), n_phi)
        self.data_shape = (self.n_q, n_phi)
        self.harmonic_padding = harmonic_padding
        self.miller_margin = miller_margin
        self.q_block_size = min(q_block_size, self.n_q)
        self.max_cutoff = max_cutoff
        self.derivative_max_order = derivative_max_order

        self.r_centers = torch.as_tensor(
            r_np, dtype=self.real_dtype, device=self.device
        )
        self.z_centers = torch.as_tensor(
            z_np, dtype=self.real_dtype, device=self.device
        )
        self.default_q_perp = torch.as_tensor(
            q_perp_np, dtype=self.real_dtype, device=self.device
        )
        self.default_q_z = torch.as_tensor(
            q_z_np, dtype=self.real_dtype, device=self.device
        )
        self.support_q_perp = torch.as_tensor(
            np.ascontiguousarray(support_np),
            dtype=self.real_dtype,
            device=self.device,
        )
        self.support_cutoffs = torch.as_tensor(
            np.ascontiguousarray(cutoffs_np),
            dtype=torch.long,
            device=self.device,
        )

        signed_modes_np = np.concatenate(
            (
                np.arange(max_cutoff + 1, dtype=np.int64),
                np.arange(-max_cutoff, 0, dtype=np.int64),
            )
        )
        mode_indices_np = np.mod(signed_modes_np, n_phi)
        self.signed_modes = torch.as_tensor(
            signed_modes_np, dtype=torch.long, device=self.device
        )
        self.mode_indices = torch.as_tensor(
            mode_indices_np, dtype=torch.long, device=self.device
        )
        self.mode_abs = torch.abs(self.signed_modes)
        self.left_neighbor_abs = torch.abs(self.signed_modes - 1)
        self.right_neighbor_abs = torch.abs(self.signed_modes + 1)
        self.n_modes = int(signed_modes_np.size)

        resident = (
            self.r_centers,
            self.z_centers,
            self.default_q_perp,
            self.default_q_z,
            self.support_q_perp,
            self.support_cutoffs,
            self.signed_modes,
            self.mode_indices,
            self.mode_abs,
            self.left_neighbor_abs,
            self.right_neighbor_abs,
        )
        self.basis_mib = float(
            sum(t.nelement() * t.element_size() for t in resident)
            / (1024.0 * 1024.0)
        )

    def _object_tensor(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != self.object_shape:
            raise ValueError(f"object values must have shape {self.object_shape}")
        if not bool(self.torch.all(self.torch.isfinite(tensor)).item()):
            raise ValueError("object values must contain only finite values")
        return tensor

    def _data_tensor(self, values: Any) -> Any:
        tensor = self.torch.as_tensor(values, device=self.device).to(
            dtype=self.complex_dtype
        )
        if tuple(tensor.shape) != self.data_shape:
            raise ValueError(f"data values must have shape {self.data_shape}")
        if not bool(self.torch.all(self.torch.isfinite(tensor)).item()):
            raise ValueError("data values must contain only finite values")
        return tensor

    def _resolve_geometry(self, q_perp: Any | None, q_z: Any | None) -> tuple[Any, Any]:
        qp = self.default_q_perp if q_perp is None else self.torch.as_tensor(
            q_perp, device=self.device
        ).to(dtype=self.real_dtype)
        qz = self.default_q_z if q_z is None else self.torch.as_tensor(
            q_z, device=self.device
        ).to(dtype=self.real_dtype)
        expected = (self.n_q,)
        if tuple(qp.shape) != expected or tuple(qz.shape) != expected:
            raise ValueError(f"q_perp and q_z must have shape {expected}")
        if not bool(self.torch.all(self.torch.isfinite(qp)).item()) or not bool(
            self.torch.all(self.torch.isfinite(qz)).item()
        ):
            raise ValueError("q_perp and q_z must contain only finite values")
        if bool(self.torch.any(qp < 0.0).item()):
            raise ValueError("q_perp must be non-negative")
        required = self.torch.ceil(
            qp.detach()[:, None] * self.torch.abs(self.r_centers)[None, :]
        ).to(dtype=self.torch.long) + self.harmonic_padding
        if bool(self.torch.any(required > self.support_cutoffs).item()):
            raise ValueError(
                "q_perp exceeds the frozen support envelope; rebuild the operator "
                "with a larger support_q_perp"
            )
        return qp, qz

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
            return builder(
                q_perp.detach(),
                self.r_centers,
                n_phi=self.n_phi,
                max_cutoff=max_order,
                extra_order=self.miller_margin,
                torch=self.torch,
            )

        q_np = np.ascontiguousarray(q_perp.detach().cpu().numpy(), dtype=np.float64)
        r_np = np.ascontiguousarray(
            self.r_centers.detach().cpu().numpy(), dtype=np.float64
        )
        from .solvers import _cpp_solver_module

        cpp = _cpp_solver_module(required=False)
        function_name = (
            "analytic_kernel_hat_modes_miller64"
            if self.complex_dtype == self.torch.complex64
            else "analytic_kernel_hat_modes_miller"
        )
        if cpp is not None and hasattr(cpp, function_name):
            positive_np = np.asarray(
                getattr(cpp, function_name)(
                    q_np,
                    r_np,
                    self.n_phi,
                    int(max_order),
                    self.miller_margin,
                ),
                dtype=self._numpy_complex_dtype,
            )
        else:
            from scipy import special

            orders = np.arange(max_order + 1, dtype=np.int64)
            phase = np.power(1j, orders).astype(self._numpy_complex_dtype)
            positive_np = (
                self.n_phi
                * special.jv(
                    orders[None, None, :],
                    q_np[:, None, None] * r_np[None, :, None],
                )
                * phase[None, None, :]
            ).astype(self._numpy_complex_dtype, copy=False)
        return self.torch.as_tensor(
            np.ascontiguousarray(positive_np),
            dtype=self.complex_dtype,
            device=self.device,
        )

    def _z_phase(self, q_z: Any) -> Any:
        return self.torch.exp(
            1j * q_z[:, None] * self.z_centers[None, :]
        ).to(dtype=self.complex_dtype)

    def _mode_mask(self, start: int, stop: int) -> Any:
        return self.mode_abs[None, None, :] <= self.support_cutoffs[
            start:stop, :, None
        ]

    def _fill_kernel_buffer(
        self,
        positive: Any,
        indices: Any,
        mask: Any,
        buffer: Any,
    ) -> Any:
        self.torch.index_select(positive, 2, indices, out=buffer)
        buffer.masked_fill_(self.torch.logical_not(mask), 0)
        return buffer.permute(2, 0, 1)

    def _selected_object_modes(self, values: Any) -> Any:
        return self.torch.fft.fft(values, dim=-1).index_select(
            -1, self.mode_indices
        )

    def _axial_contraction(self, z_phase: Any, object_by_mode: Any) -> Any:
        return self.torch.bmm(
            z_phase.unsqueeze(0).expand(self.n_modes, -1, -1),
            object_by_mode,
        )

    def _forward_fourier_resolved(self, values: Any, q_perp: Any, q_z: Any) -> Any:
        object_modes = self._selected_object_modes(values)
        object_by_mode = object_modes.permute(2, 1, 0).contiguous()
        z_phase = self._z_phase(q_z)
        selected = self.torch.empty(
            (self.n_q, self.n_modes),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            positive = self._positive_kernel(q_perp[start:stop], self.max_cutoff)
            buffer = self.torch.empty(
                (stop - start, self.object_shape[0], self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            kernel = self._fill_kernel_buffer(
                positive,
                self.mode_abs,
                self._mode_mask(start, stop),
                buffer,
            )
            axial = self._axial_contraction(
                z_phase[start:stop], object_by_mode
            )
            selected[start:stop] = self.torch.sum(axial * kernel, dim=-1).transpose(
                0, 1
            )
        return selected

    def _geometry_fourier_resolved(
        self, values: Any, q_perp: Any, q_z: Any
    ) -> tuple[Any, Any, Any]:
        object_modes = self._selected_object_modes(values)
        object_by_mode = object_modes.permute(2, 1, 0).contiguous()
        z_phase = self._z_phase(q_z)
        axial_derivative_phase = (
            1j * self.z_centers[None, :] * z_phase
        ).to(dtype=self.complex_dtype)
        outputs = [
            self.torch.empty(
                (self.n_q, self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            for _ in range(3)
        ]
        selected, derivative_perp, derivative_z = outputs
        radial_weight = self.r_centers[None, None, :]

        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            positive = self._positive_kernel(
                q_perp[start:stop], self.derivative_max_order
            )
            mask = self._mode_mask(start, stop)
            buffer = self.torch.empty(
                (stop - start, self.object_shape[0], self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            axial = self._axial_contraction(
                z_phase[start:stop], object_by_mode
            )
            axial_z = self._axial_contraction(
                axial_derivative_phase[start:stop], object_by_mode
            )
            kernel = self._fill_kernel_buffer(positive, self.mode_abs, mask, buffer)
            selected[start:stop] = self.torch.sum(axial * kernel, dim=-1).transpose(
                0, 1
            )
            derivative_z[start:stop] = self.torch.sum(
                axial_z * kernel, dim=-1
            ).transpose(0, 1)

            radial_axial = axial * radial_weight
            kernel = self._fill_kernel_buffer(
                positive, self.left_neighbor_abs, mask, buffer
            )
            radial_sum = self.torch.sum(radial_axial * kernel, dim=-1).transpose(
                0, 1
            )
            kernel = self._fill_kernel_buffer(
                positive, self.right_neighbor_abs, mask, buffer
            )
            radial_sum.add_(
                self.torch.sum(radial_axial * kernel, dim=-1).transpose(0, 1)
            )
            derivative_perp[start:stop] = 0.5j * radial_sum
        return selected, derivative_perp, derivative_z

    def _synthesize(self, selected: Any) -> Any:
        spectrum = self.torch.zeros(
            self.data_shape, dtype=self.complex_dtype, device=self.device
        )
        spectrum.index_copy_(-1, self.mode_indices, selected)
        return self.torch.fft.ifft(spectrum, dim=-1)

    def forward(
        self,
        object_values: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> Any:
        """Evaluate the compact ACFO operator at one geometry."""

        values = self._object_tensor(object_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        return self._synthesize(self._forward_fourier_resolved(values, qp, qz))

    def forward_with_geometry_derivatives(
        self,
        object_values: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Return prediction, ``d/dq_perp`` and ``d/dq_z`` point samples."""

        values = self._object_tensor(object_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        selected, derivative_perp, derivative_z = self._geometry_fourier_resolved(
            values, qp, qz
        )
        return tuple(
            self._synthesize(value)
            for value in (selected, derivative_perp, derivative_z)
        )

    def _jet_fourier_resolved(
        self,
        values: Any,
        q_perp: Any,
        q_z: Any,
        max_total_order: int,
    ) -> dict[tuple[int, int, int], Any]:
        """Return normalized mixed geometry derivatives in Fourier modes.

        A key ``(a, b, c)`` stores

        ``d^(a+b+c) F / (dq_perp^a dq_z^b dphi^c) / (a! b! c!)``.

        The radial recurrence for ``K_m = n_phi i^m J_m`` is

        ``K_m^(a) = (i/2)^a sum_j binom(a,j) K_(m-a+2j)``.
        """

        torch = self.torch
        object_modes = self._selected_object_modes(values)
        object_by_mode = object_modes.permute(2, 1, 0).contiguous()
        z_phase = self._z_phase(q_z)
        mode_real = self.signed_modes.to(dtype=self.real_dtype)
        keys = [
            (a, b, c)
            for total in range(max_total_order + 1)
            for a in range(total + 1)
            for b in range(total - a + 1)
            for c in (total - a - b,)
        ]
        outputs = {
            key: torch.empty(
                (self.n_q, self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            for key in keys
        }

        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            positive = self._positive_kernel(
                q_perp[start:stop], self.max_cutoff + max_total_order
            )
            mask = self._mode_mask(start, stop)
            buffer = torch.empty(
                (stop - start, self.object_shape[0], self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )

            axial_by_order = []
            for b in range(max_total_order + 1):
                normalized_phase = z_phase[start:stop] * (
                    (1j * self.z_centers[None, :]) ** b / math.factorial(b)
                )
                axial_by_order.append(
                    self._axial_contraction(normalized_phase, object_by_mode)
                )

            radial_by_order = []
            for a in range(max_total_order + 1):
                radial = torch.zeros(
                    (self.n_modes, stop - start, self.object_shape[0]),
                    dtype=self.complex_dtype,
                    device=self.device,
                )
                for j in range(a + 1):
                    neighbor_abs = torch.abs(self.signed_modes - a + 2 * j)
                    kernel = self._fill_kernel_buffer(
                        positive, neighbor_abs, mask, buffer
                    )
                    radial.add_(math.comb(a, j) * kernel)
                radial.mul_((0.5j) ** a / math.factorial(a))
                radial.mul_(self.r_centers[None, None, :] ** a)
                radial_by_order.append(radial)

            for a, b, c in keys:
                selected = torch.sum(
                    axial_by_order[b] * radial_by_order[a], dim=-1
                ).transpose(0, 1)
                if c:
                    selected = selected * (
                        (1j * mode_real) ** c / math.factorial(c)
                    )[None, :]
                outputs[(a, b, c)][start:stop] = selected
        return outputs

    def forward_jet(
        self,
        object_values: Any,
        max_total_order: int,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> dict[tuple[int, int, int], Any]:
        """Return a normalized ``(q_perp, q_z, phi)`` Taylor jet.

        The output contains every mixed derivative whose total order is at
        most ``max_total_order``.  Integer harmonic support remains frozen;
        the construction additionally requires ``H + max_total_order`` to
        stay below angular Nyquist.
        """

        max_total_order = int(max_total_order)
        if max_total_order < 0:
            raise ValueError("max_total_order must be non-negative")
        if self.max_cutoff + max_total_order >= self.n_phi // 2:
            raise ValueError(
                "Taylor-jet harmonic support reaches angular Nyquist; increase "
                "n_phi so H + max_total_order < n_phi / 2"
            )
        values = self._object_tensor(object_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        selected = self._jet_fourier_resolved(
            values, qp, qz, max_total_order
        )
        return {key: self._synthesize(value) for key, value in selected.items()}

    def evaluate_jet(
        self,
        jet: dict[tuple[int, int, int], Any],
        delta_q_perp: Any = 0.0,
        delta_q_z: Any = 0.0,
        delta_phi: Any = 0.0,
        *,
        max_total_order: int | None = None,
    ) -> Any:
        """Evaluate a normalized Taylor jet at one per-target displacement."""

        if not jet or (0, 0, 0) not in jet:
            raise ValueError("jet must contain the zeroth-order coefficient")

        def displacement(value: Any, name: str) -> Any:
            tensor = self.torch.as_tensor(
                value, dtype=self.real_dtype, device=self.device
            )
            if tensor.ndim == 0:
                tensor = tensor.expand(self.n_q)
            if tuple(tensor.shape) != (self.n_q,) or not bool(
                self.torch.all(self.torch.isfinite(tensor)).item()
            ):
                raise ValueError(f"{name} must be scalar or have shape ({self.n_q},)")
            return tensor

        d_perp = displacement(delta_q_perp, "delta_q_perp")
        d_z = displacement(delta_q_z, "delta_q_z")
        d_phi = displacement(delta_phi, "delta_phi")
        available_order = max(sum(key) for key in jet)
        order = available_order if max_total_order is None else int(max_total_order)
        if order < 0 or order > available_order:
            raise ValueError("max_total_order is outside the available jet")
        output = self.torch.zeros_like(jet[(0, 0, 0)])
        for (a, b, c), coefficient in jet.items():
            if a + b + c <= order:
                output = output + coefficient * (
                    d_perp**a * d_z**b * d_phi**c
                )[:, None]
        return output

    def geometry_jvp(
        self,
        object_values: Any,
        delta_q_perp: Any,
        delta_q_z: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> Any:
        """Apply the frozen-support real geometry Jacobian to one direction."""

        values = self._object_tensor(object_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        delta_perp = self.torch.as_tensor(
            delta_q_perp, dtype=self.real_dtype, device=self.device
        )
        delta_z = self.torch.as_tensor(
            delta_q_z, dtype=self.real_dtype, device=self.device
        )
        if tuple(delta_perp.shape) != (self.n_q,) or tuple(delta_z.shape) != (
            self.n_q,
        ):
            raise ValueError(f"geometry directions must have shape ({self.n_q},)")
        _, derivative_perp, derivative_z = self._geometry_fourier_resolved(
            values, qp, qz
        )
        return self._synthesize(
            derivative_perp * delta_perp[:, None]
            + derivative_z * delta_z[:, None]
        )

    def _adjoint_resolved(self, data_values: Any, q_perp: Any, q_z: Any) -> Any:
        data_modes = self.torch.fft.fft(data_values, dim=-1).index_select(
            -1, self.mode_indices
        )
        z_phase = self._z_phase(q_z)
        selected_by_mode = self.torch.zeros(
            (self.n_modes, self.object_shape[0], self.object_shape[1]),
            dtype=self.complex_dtype,
            device=self.device,
        )
        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            positive = self._positive_kernel(q_perp[start:stop], self.max_cutoff)
            buffer = self.torch.empty(
                (stop - start, self.object_shape[0], self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            kernel = self._fill_kernel_buffer(
                positive,
                self.mode_abs,
                self._mode_mask(start, stop),
                buffer,
            )
            weighted = data_modes[start:stop].transpose(0, 1).unsqueeze(
                -1
            ) * self.torch.conj(z_phase[start:stop]).unsqueeze(0)
            selected_by_mode.add_(
                self.torch.bmm(self.torch.conj(kernel).transpose(1, 2), weighted)
            )
        object_spectrum = self.torch.zeros(
            self.object_shape, dtype=self.complex_dtype, device=self.device
        )
        object_spectrum.index_copy_(
            -1, self.mode_indices, selected_by_mode.permute(1, 2, 0)
        )
        return self.torch.fft.ifft(object_spectrum, dim=-1)

    def adjoint(
        self,
        data_values: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> Any:
        """Apply the Euclidean adjoint paired with :meth:`forward`."""

        data = self._data_tensor(data_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        return self._adjoint_resolved(data, qp, qz)

    def geometry_vjp(
        self,
        object_values: Any,
        data_cotangent: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> tuple[Any, Any]:
        """Apply the real geometry VJP without derivative inverse FFTs."""

        values = self._object_tensor(object_values)
        cotangent = self._data_tensor(data_cotangent)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        _, derivative_perp, derivative_z = self._geometry_fourier_resolved(
            values, qp, qz
        )
        cotangent_fourier = self.torch.fft.fft(cotangent, dim=-1).index_select(
            -1, self.mode_indices
        )
        scale = 1.0 / self.n_phi
        gradient_perp = self.torch.real(
            self.torch.sum(
                self.torch.conj(cotangent_fourier) * derivative_perp, dim=-1
            )
        ) * scale
        gradient_z = self.torch.real(
            self.torch.sum(
                self.torch.conj(cotangent_fourier) * derivative_z, dim=-1
            )
        ) * scale
        return gradient_perp, gradient_z

    def _fused_adjoint_geometry_vjp(
        self,
        object_values: Any,
        data_cotangent: Any,
        q_perp: Any,
        q_z: Any,
        *,
        need_object_gradient: bool,
        need_q_perp_gradient: bool,
        need_q_z_gradient: bool,
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Apply the prepared first-order pullback in one streamed pass.

        The unfused custom backward evaluates the object adjoint and geometry
        VJP independently.  That repeats the data FFT, axial phase, q-block
        dispatch and Miller recurrence.  This routine prepares each bounded
        ``H + 1`` Miller block once and consumes it for every requested action:

        * the Euclidean object adjoint,
        * the two neighboring-order radial contractions, and
        * the axial derivative contraction.

        Geometry gradients are reduced against the cotangent inside the q
        block.  Full ``(n_q, n_modes)`` derivative arrays are never retained.
        Integer support selection remains outside the differentiable map.
        """

        if not (
            need_object_gradient
            or need_q_perp_gradient
            or need_q_z_gradient
        ):
            return None, None, None

        torch = self.torch
        # This is an internal compiled action.  Public entry points validate
        # object/data shapes before dispatch, while autograd supplies a
        # correctly shaped cotangent.  Revalidating here would introduce GPU
        # synchronization inside every backward call.
        values = object_values
        cotangent = data_cotangent.to(
            dtype=self.complex_dtype, device=self.device
        )
        data_modes = torch.fft.fft(cotangent, dim=-1).index_select(
            -1, self.mode_indices
        )
        z_phase = self._z_phase(q_z)

        need_geometry = need_q_perp_gradient or need_q_z_gradient
        object_by_mode = None
        if need_geometry:
            object_modes = self._selected_object_modes(values)
            object_by_mode = object_modes.permute(2, 1, 0).contiguous()

        selected_by_mode = None
        if need_object_gradient:
            selected_by_mode = torch.zeros(
                (self.n_modes, self.object_shape[0], self.object_shape[1]),
                dtype=self.complex_dtype,
                device=self.device,
            )

        gradient_perp = (
            torch.empty(self.n_q, dtype=self.real_dtype, device=self.device)
            if need_q_perp_gradient
            else None
        )
        gradient_z = (
            torch.empty(self.n_q, dtype=self.real_dtype, device=self.device)
            if need_q_z_gradient
            else None
        )
        radial_weight = self.r_centers[None, None, :]
        scale = 1.0 / self.n_phi
        max_order = (
            self.derivative_max_order if need_q_perp_gradient else self.max_cutoff
        )

        for start in range(0, self.n_q, self.q_block_size):
            stop = min(start + self.q_block_size, self.n_q)
            positive = self._positive_kernel(q_perp[start:stop], max_order)
            mask = self._mode_mask(start, stop)
            buffer = torch.empty(
                (stop - start, self.object_shape[0], self.n_modes),
                dtype=self.complex_dtype,
                device=self.device,
            )
            kernel = self._fill_kernel_buffer(
                positive, self.mode_abs, mask, buffer
            )
            cotangent_block = data_modes[start:stop]

            if selected_by_mode is not None:
                weighted = cotangent_block.transpose(0, 1).unsqueeze(
                    -1
                ) * torch.conj(z_phase[start:stop]).unsqueeze(0)
                selected_by_mode.add_(
                    torch.bmm(torch.conj(kernel).transpose(1, 2), weighted)
                )

            if need_geometry:
                assert object_by_mode is not None
                axial = self._axial_contraction(
                    z_phase[start:stop], object_by_mode
                )

                if gradient_z is not None:
                    axial_z = self._axial_contraction(
                        (
                            1j
                            * self.z_centers[None, :]
                            * z_phase[start:stop]
                        ).to(dtype=self.complex_dtype),
                        object_by_mode,
                    )
                    derivative_z = torch.sum(axial_z * kernel, dim=-1).transpose(
                        0, 1
                    )
                    gradient_z[start:stop] = torch.real(
                        torch.sum(
                            torch.conj(cotangent_block) * derivative_z, dim=-1
                        )
                    ) * scale

                if gradient_perp is not None:
                    radial_axial = axial * radial_weight
                    kernel = self._fill_kernel_buffer(
                        positive, self.left_neighbor_abs, mask, buffer
                    )
                    radial_sum = torch.sum(
                        radial_axial * kernel, dim=-1
                    ).transpose(0, 1)
                    kernel = self._fill_kernel_buffer(
                        positive, self.right_neighbor_abs, mask, buffer
                    )
                    radial_sum.add_(
                        torch.sum(radial_axial * kernel, dim=-1).transpose(0, 1)
                    )
                    derivative_perp = 0.5j * radial_sum
                    gradient_perp[start:stop] = torch.real(
                        torch.sum(
                            torch.conj(cotangent_block) * derivative_perp,
                            dim=-1,
                        )
                    ) * scale

        object_gradient = None
        if selected_by_mode is not None:
            object_spectrum = torch.zeros(
                self.object_shape,
                dtype=self.complex_dtype,
                device=self.device,
            )
            object_spectrum.index_copy_(
                -1, self.mode_indices, selected_by_mode.permute(1, 2, 0)
            )
            object_gradient = torch.fft.ifft(object_spectrum, dim=-1)
        return object_gradient, gradient_perp, gradient_z

    def fused_pullback(
        self,
        object_values: Any,
        data_cotangent: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
        *,
        object_gradient: bool = True,
        geometry_gradient: bool = True,
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Return a prepared fused object/geometry pullback.

        This explicit API is useful for matrix-free optimizers that already
        own the data cotangent and do not need PyTorch to build an autograd
        graph around the operator.
        """

        values = self._object_tensor(object_values)
        cotangent = self._data_tensor(data_cotangent)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        return self._fused_adjoint_geometry_vjp(
            values,
            cotangent,
            qp,
            qz,
            need_object_gradient=bool(object_gradient),
            need_q_perp_gradient=bool(geometry_gradient),
            need_q_z_gradient=bool(geometry_gradient),
        )

    def _autograd_forward_impl(
        self,
        object_values: Any,
        q_perp: Any | None,
        q_z: Any | None,
        *,
        fused_backward: bool,
    ) -> Any:
        """Shared custom-autograd implementation for benchmarkable dispatch."""

        values = self._object_tensor(object_values)
        qp, qz = self._resolve_geometry(q_perp, q_z)
        plan = self
        torch = self.torch

        class _AnalyticAcfoFunction(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                object_tensor: Any,
                q_perp_tensor: Any,
                q_z_tensor: Any,
            ) -> Any:
                ctx.save_for_backward(object_tensor, q_perp_tensor, q_z_tensor)
                selected = plan._forward_fourier_resolved(
                    object_tensor, q_perp_tensor, q_z_tensor
                )
                return plan._synthesize(selected)

            @staticmethod
            def backward(ctx: Any, data_cotangent: Any) -> tuple[Any, Any, Any]:
                object_tensor, q_perp_tensor, q_z_tensor = ctx.saved_tensors
                if fused_backward:
                    return plan._fused_adjoint_geometry_vjp(
                        object_tensor,
                        data_cotangent.to(dtype=plan.complex_dtype),
                        q_perp_tensor,
                        q_z_tensor,
                        need_object_gradient=bool(ctx.needs_input_grad[0]),
                        need_q_perp_gradient=bool(ctx.needs_input_grad[1]),
                        need_q_z_gradient=bool(ctx.needs_input_grad[2]),
                    )

                object_gradient = None
                gradient_perp = None
                gradient_z = None
                if ctx.needs_input_grad[0]:
                    object_gradient = plan._adjoint_resolved(
                        data_cotangent.to(dtype=plan.complex_dtype),
                        q_perp_tensor,
                        q_z_tensor,
                    )
                if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                    gradient_perp, gradient_z = plan.geometry_vjp(
                        object_tensor,
                        data_cotangent,
                        q_perp_tensor,
                        q_z_tensor,
                    )
                    if not ctx.needs_input_grad[1]:
                        gradient_perp = None
                    if not ctx.needs_input_grad[2]:
                        gradient_z = None
                return object_gradient, gradient_perp, gradient_z

        return _AnalyticAcfoFunction.apply(values, qp, qz)

    def autograd_forward(
        self,
        object_values: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> Any:
        """Evaluate with the fused analytic first-order PyTorch backward rule.

        Backward streams one bounded Miller block through the object adjoint
        and requested geometry pullbacks.  It avoids a special-function graph,
        derivative kernel table and duplicate object/geometry operator passes.
        """

        return self._autograd_forward_impl(
            object_values, q_perp, q_z, fused_backward=True
        )

    def autograd_forward_unfused(
        self,
        object_values: Any,
        q_perp: Any | None = None,
        q_z: Any | None = None,
    ) -> Any:
        """Evaluate with the historical two-pass analytic backward.

        This method is retained as a matched implementation baseline for
        correctness, timing and memory audits.  New code should use
        :meth:`autograd_forward`.
        """

        return self._autograd_forward_impl(
            object_values, q_perp, q_z, fused_backward=False
        )

    def synchronize(self) -> None:
        """Synchronize the selected CUDA device, if present."""

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
