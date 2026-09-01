"""GPU-resident fused Fourier and polarization stages for flat-detector SANS.

This module deliberately implements the geometry used by NuMagSANS: the
incident beam is the cylindrical orbit axis and every detector row is a full
azimuthal ring with ``q_z == 0`` in the ACFO frame.  Histogram construction is
still a setup operation performed by :class:`PreparedBandlimitedMagneticSansOperator`;
the repeated Fourier contraction, inverse FFT and twelve detector observables
remain on the selected torch device.

Torch is an optional dependency.  Importing this module does not import torch;
it is supplied explicitly when a plan is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .magnetic_sans import PreparedBandlimitedMagneticSansOperator
from .solvers import _cpp_solver_module


@dataclass(frozen=True)
class TorchMagneticSansAmplitudes:
    """Four GPU-resident Fourier channels on a flat polar detector."""

    nuclear: Any
    magnetization: Any


def _torch_complex_dtype(torch: Any, dtype: str) -> tuple[Any, np.dtype]:
    if dtype == "complex64":
        return torch.complex64, np.dtype(np.complex64)
    if dtype == "complex128":
        return torch.complex128, np.dtype(np.complex128)
    raise ValueError("dtype must be 'complex64' or 'complex128'")


class TorchFlatDetectorBandlimitedMagneticSansOperator:
    """Fused four-channel ACFO contraction for full flat-detector rings.

    The source histograms must be real.  This is the physically relevant
    NuMagSANS contract and lets the implementation store only modes ``0..H``;
    the negative Fourier modes are reconstructed by conjugate symmetry before
    a single batched inverse FFT.
    """

    def __init__(
        self,
        prepared: PreparedBandlimitedMagneticSansOperator,
        *,
        torch: Any,
        device: Any,
        dtype: str = "complex64",
        detector_directions: Any | None = None,
        source_deposition: str = "nearest",
        kernel_backend: str = "auto",
        active_r_indices_override: Any | None = None,
    ) -> None:
        if not np.allclose(prepared.manifold.q_z, 0.0, rtol=0.0, atol=1e-14):
            raise ValueError("flat-detector torch operator requires q_z == 0")
        if len(prepared.channel_plans) != 4:
            raise ValueError("exactly four nuclear/magnetization channels are required")
        template = prepared.channel_plans[0]
        if any(plan.binned.hist.shape != template.binned.hist.shape for plan in prepared.channel_plans):
            raise ValueError("all channel histograms must share one cylindrical grid")
        if any(np.iscomplexobj(plan.binned.hist) for plan in prepared.channel_plans):
            raise ValueError("the compact torch path requires real source histograms")
        if template.form_factors.shape[0] != 1:
            raise ValueError("the magnetic-SANS torch path currently requires one source species")

        self.torch = torch
        self.device = device
        self.complex_dtype, np_complex = _torch_complex_dtype(torch, dtype)
        self.real_dtype = torch.float32 if self.complex_dtype == torch.complex64 else torch.float64
        self.phase_sign = int(prepared.phase_sign)
        self.n_phi = int(template.binned.n_phi)
        self.n_half = int(template.n_half)
        self.n_q = int(prepared.manifold.q_perp.size)
        self.n_r = int(template.binned.r_centers.size)
        self._r_edges = np.array(template.binned.r_edges, copy=True)
        self._z_edges = np.array(template.binned.z_edges, copy=True)
        self.margin = int(prepared.margin)
        self.cutoff_bin_size = int(prepared.cutoff_bin_size)
        if source_deposition not in {"nearest", "linear"}:
            raise ValueError("source_deposition must be 'nearest' or 'linear'")
        self.source_deposition = str(source_deposition)
        if kernel_backend not in {"auto", "cpu_miller", "gpu_miller"}:
            raise ValueError(
                "kernel_backend must be 'auto', 'cpu_miller', or 'gpu_miller'"
            )
        self.requested_kernel_backend = str(kernel_backend)

        q_indices = np.arange(self.n_q, dtype=np.intp)
        points = np.asarray(prepared.source_points, dtype=np.float64)
        radius = np.hypot(points[:, 0], points[:, 1])
        if self.source_deposition == "nearest":
            radial_indices = (
                radius * (self.n_r / float(self._r_edges[-1]))
            ).astype(np.int64)
            np.clip(radial_indices, 0, self.n_r - 1, out=radial_indices)
            active_r_indices = np.unique(radial_indices).astype(
                np.intp, copy=False
            )
        else:
            lower, upper, fraction = self._linear_center_indices(
                radius,
                step=float(self._r_edges[-1]) / self.n_r,
                size=self.n_r,
                periodic=False,
            )
            active_r_indices = np.unique(
                np.concatenate([lower, upper])
            ).astype(np.intp, copy=False)
        if active_r_indices_override is not None:
            override = np.asarray(active_r_indices_override, dtype=np.intp)
            if override.ndim != 1 or override.size == 0:
                raise ValueError(
                    "active_r_indices_override must be a nonempty one-dimensional array"
                )
            if np.any(override < 0) or np.any(override >= self.n_r):
                raise ValueError(
                    "active_r_indices_override contains an index outside the radial grid"
                )
            if np.unique(override).size != override.size or np.any(np.diff(override) <= 0):
                raise ValueError(
                    "active_r_indices_override must be sorted and contain unique indices"
                )
            active_r_indices = override
        self.active_r_indices_were_overridden = active_r_indices_override is not None
        if active_r_indices.size == 0:
            raise ValueError("at least one occupied cylindrical radius is required")
        self.active_r_indices = np.ascontiguousarray(active_r_indices)
        self.n_active_r = int(active_r_indices.size)

        cutoffs = template._r_dependent_cutoff_matrix(
            q_indices,
            margin=self.margin,
            cutoff_bin_size=self.cutoff_bin_size,
        )
        active_cutoffs = np.ascontiguousarray(cutoffs[:, active_r_indices])
        max_h = int(np.max(active_cutoffs))
        if max_h >= self.n_half:
            raise ValueError(
                "required harmonic cutoff reaches angular Nyquist; increase n_phi"
            )
        self.max_h = max_h
        cpp_solvers = _cpp_solver_module(required=False)
        miller_name = (
            "analytic_kernel_hat_modes_miller64"
            if self.complex_dtype == torch.complex64
            else "analytic_kernel_hat_modes_miller"
        )
        cpu_miller_available = cpp_solvers is not None and callable(
            getattr(cpp_solvers, miller_name, None)
        )
        self.cpu_miller_reference_available = bool(cpu_miller_available)
        if kernel_backend == "gpu_miller":
            if self.complex_dtype != torch.complex64:
                raise ValueError("GPU Miller prototype currently requires complex64")
            self.kernel_backend = "gpu_miller_downward_recurrence"
            self.miller_recurrence_margin = 64
        elif cpu_miller_available:
            self.kernel_backend = "cpp_miller_downward_recurrence"
            self.miller_recurrence_margin = 64
        else:
            if kernel_backend == "cpu_miller":
                raise RuntimeError("compiled CPU Miller kernel is unavailable")
            self.kernel_backend = "scipy_special_jv_fallback"
            self.miller_recurrence_margin = None
        modes = np.arange(max_h + 1, dtype=np.int64)
        if kernel_backend == "gpu_miller":
            from .gpu_miller import gpu_miller_kernel64

            kernel_gpu = gpu_miller_kernel64(
                template.q_perp[q_indices],
                template.binned.r_centers[active_r_indices],
                n_phi=self.n_phi,
                max_cutoff=max_h,
                extra_order=64,
                torch=torch,
            )
            kernel_gpu *= torch.as_tensor(
                modes[None, None, :] <= active_cutoffs[:, :, None],
                dtype=self.complex_dtype,
                device=device,
            )
        else:
            kernel = template._analytic_kernel_hat_modes_r(
                q_indices,
                active_r_indices,
                max_h,
            ).astype(np_complex, copy=False)
            kernel *= modes[None, None, :] <= active_cutoffs[:, :, None]

        # The archived detector has only a few rounded harmonic tiers.  Store
        # one compact kernel per tier so the hot contraction does not multiply
        # the masked high-h tail.  This is algebraically identical to the old
        # dense zero-masked einsum and preserves arbitrary q ordering.
        row_cutoffs = np.max(active_cutoffs, axis=1)
        kernel_tiers = []
        for h_limit in np.unique(row_cutoffs):
            q_local = np.flatnonzero(row_cutoffs == h_limit).astype(np.int64)
            q_tensor = torch.as_tensor(q_local, dtype=torch.long, device=device)
            if kernel_backend == "gpu_miller":
                local_kernel = kernel_gpu.index_select(0, q_tensor)[
                    :, :, : int(h_limit) + 1
                ].contiguous()
            else:
                local_kernel = torch.as_tensor(
                    np.ascontiguousarray(
                        kernel[q_local, :, : int(h_limit) + 1]
                    ),
                    dtype=self.complex_dtype,
                    device=device,
                )
            kernel_tiers.append((q_tensor, local_kernel))
        self._kernel_tiers = tuple(kernel_tiers)
        self._tier_cutoffs = tuple(
            int(local_kernel.shape[-1] - 1)
            for _, local_kernel in self._kernel_tiers
        )
        self.dense_half_contraction_terms = int(
            self.n_q * self.n_r * (self.max_h + 1)
        )
        self.compact_half_contraction_terms = int(
            sum(
                int(q_local.numel())
                * self.n_active_r
                * int(local_kernel.shape[-1])
                for q_local, local_kernel in self._kernel_tiers
            )
        )

        # q_z is zero and the single form factor is one, so the axial and
        # species contractions reduce exactly to a sum before the hot loop.
        # Keep only geometrically occupied radii; this remains valid for all
        # later weight updates on the same fixed point set.
        source = np.stack(
            [
                np.sum(
                    plan.hhat_half_modes(max_h),
                    axis=(0, 2),
                    dtype=np_complex,
                )[active_r_indices]
                for plan in prepared.channel_plans
            ],
            axis=0,
        )
        self.source_crh = torch.as_tensor(
            np.ascontiguousarray(source, dtype=np_complex),
            dtype=self.complex_dtype,
            device=device,
        )
        self._positive_slots = torch.arange(max_h + 1, device=device)
        if max_h:
            self._negative_slots = torch.arange(
                self.n_phi - max_h,
                self.n_phi,
                device=device,
            )
        else:
            self._negative_slots = None

        if detector_directions is None:
            phi = np.asarray(prepared.phi, dtype=np.float64)
            directions = np.stack(
                [np.cos(phi), np.sin(phi), np.zeros_like(phi)],
                axis=-1,
            )
        else:
            directions = np.asarray(detector_directions, dtype=np.float64)
            if directions.shape != (self.n_phi, 3):
                raise ValueError(
                    f"detector_directions must have shape ({self.n_phi}, 3)"
                )
            if not np.all(np.isfinite(directions)):
                raise ValueError("detector_directions must be finite")
            norms = np.linalg.norm(directions, axis=-1)
            if np.any(norms <= 0.0):
                raise ValueError("detector_directions must be nonzero")
            directions = directions / norms[:, None]
        self.q_hat_phi3 = torch.as_tensor(
            directions,
            dtype=self.real_dtype,
            device=device,
        )
        self._source_flat_indices = None
        self._source_deposition_weights = None
        self._radial_compact_map = np.full(self.n_r, -1, dtype=np.int64)
        self._radial_compact_map[active_r_indices] = np.arange(
            self.n_active_r, dtype=np.int64
        )
        self._histogram_size = self.n_active_r * self.n_phi
        self.n_z = int(template.binned.z_centers.size)

    @staticmethod
    def _linear_center_indices(
        values: np.ndarray,
        *,
        step: float,
        size: int,
        periodic: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return two centre-grid indices and the upper interpolation weight."""

        scaled = np.asarray(values, dtype=np.float64) / float(step) - 0.5
        lower_raw = np.floor(scaled).astype(np.int64)
        fraction = scaled - lower_raw
        if periodic:
            lower = np.mod(lower_raw, size)
            upper = np.mod(lower_raw + 1, size)
            return lower, upper, fraction
        lower = np.array(lower_raw, copy=True)
        below = lower < 0
        above = lower >= size - 1
        lower[below] = 0
        fraction[below] = 0.0
        lower[above] = size - 1
        fraction[above] = 0.0
        upper = np.minimum(lower + 1, size - 1)
        return lower, upper, fraction

    @property
    def resident_bytes(self) -> int:
        tensors = [
            *(kernel for _, kernel in self._kernel_tiers),
            *(indices for indices, _ in self._kernel_tiers),
            self.source_crh,
            self.q_hat_phi3,
        ]
        if self._source_flat_indices is not None:
            tensors.append(self._source_flat_indices)
        if self._source_deposition_weights is not None:
            tensors.append(self._source_deposition_weights)
        return int(sum(item.nelement() * item.element_size() for item in tensors))

    @property
    def optimization_stats(self) -> dict[str, Any]:
        """Return inspectable exact-compaction statistics for this plan."""

        return {
            "dense_radii": self.n_r,
            "active_radii": self.n_active_r,
            "harmonic_tiers": self._tier_cutoffs,
            "kernel_backend": self.kernel_backend,
            "requested_kernel_backend": self.requested_kernel_backend,
            "miller_recurrence_margin": self.miller_recurrence_margin,
            "cpu_miller_reference_available": self.cpu_miller_reference_available,
            "dense_half_contraction_terms": self.dense_half_contraction_terms,
            "compact_half_contraction_terms": self.compact_half_contraction_terms,
            "contraction_term_reduction": (
                self.dense_half_contraction_terms
                / max(self.compact_half_contraction_terms, 1)
            ),
            "dense_update_state_values": 4 * self.n_r * self.n_z * self.n_phi,
            "compact_update_state_values": 4 * self.n_active_r * self.n_phi,
            "source_deposition": self.source_deposition,
            "active_r_indices_overridden": self.active_r_indices_were_overridden,
            "source_deposits_per_point": (
                1 if self.source_deposition == "nearest" else 4
            ),
        }

    def configure_source_updates(self, points: Any) -> None:
        """Cache the fixed point-to-cylinder map for changing field weights."""

        coords = np.asarray(points, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        # Geometry axes are recoverable from the prepared kernel's dimensions;
        # save their numerical edges during construction on first use.
        # These attributes are intentionally private because the update map is
        # an implementation detail, not a new geometry contract.
        if not hasattr(self, "_r_edges") or not hasattr(self, "_z_edges"):
            raise RuntimeError("source-update geometry was not retained")
        r_max = float(self._r_edges[-1])
        radius = np.hypot(coords[:, 0], coords[:, 1])
        beta = np.arctan2(coords[:, 1], coords[:, 0])
        beta[beta < 0.0] += 2.0 * np.pi
        if self.source_deposition == "nearest":
            r_idx = (radius * self.n_r / r_max).astype(np.int64)
            beta_idx = (beta * self.n_phi / (2.0 * np.pi)).astype(np.int64)
            np.clip(r_idx, 0, self.n_r - 1, out=r_idx)
            np.clip(beta_idx, 0, self.n_phi - 1, out=beta_idx)
            compact_r_idx = self._radial_compact_map[r_idx]
            if np.any(compact_r_idx < 0):
                raise ValueError(
                    "source update contains a cylindrical radius outside the prepared geometry"
                )
            flat = compact_r_idx * self.n_phi + beta_idx
            self._source_flat_indices = self.torch.as_tensor(
                np.ascontiguousarray(flat),
                dtype=self.torch.long,
                device=self.device,
            )
            self._source_deposition_weights = None
            return

        r0, r1, rf = self._linear_center_indices(
            radius,
            step=r_max / self.n_r,
            size=self.n_r,
            periodic=False,
        )
        b0, b1, bf = self._linear_center_indices(
            beta,
            step=2.0 * np.pi / self.n_phi,
            size=self.n_phi,
            periodic=True,
        )
        compact_r0 = self._radial_compact_map[r0]
        compact_r1 = self._radial_compact_map[r1]
        if np.any(compact_r0 < 0) or np.any(compact_r1 < 0):
            raise ValueError(
                "linear source update touches a radius outside the prepared geometry"
            )
        flat = np.stack(
            [
                compact_r0 * self.n_phi + b0,
                compact_r0 * self.n_phi + b1,
                compact_r1 * self.n_phi + b0,
                compact_r1 * self.n_phi + b1,
            ],
            axis=0,
        )
        deposition_weights = np.stack(
            [
                (1.0 - rf) * (1.0 - bf),
                (1.0 - rf) * bf,
                rf * (1.0 - bf),
                rf * bf,
            ],
            axis=0,
        )
        self._source_flat_indices = self.torch.as_tensor(
            np.ascontiguousarray(flat), dtype=self.torch.long, device=self.device
        )
        self._source_deposition_weights = self.torch.as_tensor(
            np.ascontiguousarray(deposition_weights),
            dtype=self.real_dtype,
            device=self.device,
        )

    def source_spectrum_from_weights(self, channel_weights: Any) -> Any:
        """Histogram and angular-FFT one changing four-channel field state."""

        if self._source_flat_indices is None:
            raise RuntimeError("call configure_source_updates(points) first")
        weights = self.torch.as_tensor(
            channel_weights,
            dtype=self.real_dtype,
            device=self.device,
        )
        source_count = int(self._source_flat_indices.shape[-1])
        if weights.ndim != 2 or weights.shape != (4, source_count):
            raise ValueError("channel_weights must have shape (4, N)")
        hist = self.torch.zeros(
            (4, self._histogram_size), dtype=self.real_dtype, device=self.device
        )
        if self._source_deposition_weights is None:
            hist.scatter_add_(
                1,
                self._source_flat_indices.unsqueeze(0).expand(4, -1),
                weights,
            )
        else:
            for local_indices, local_weights in zip(
                self._source_flat_indices,
                self._source_deposition_weights,
                strict=True,
            ):
                hist.scatter_add_(
                    1,
                    local_indices.unsqueeze(0).expand(4, -1),
                    weights * local_weights.unsqueeze(0),
                )
        hist = hist.reshape(4, self.n_active_r, self.n_phi)
        source = self.torch.fft.rfft(hist, dim=-1)[..., : self.max_h + 1]
        return source

    def _amplitudes_for_source(self, source_crh: Any) -> TorchMagneticSansAmplitudes:
        spectrum = self.torch.zeros(
            (4, self.n_q, self.n_phi),
            dtype=self.complex_dtype,
            device=self.device,
        )
        paired_source = self.torch.cat((source_crh, self.torch.conj(source_crh)), dim=0)
        for q_local, local_kernel in self._kernel_tiers:
            width = int(local_kernel.shape[-1])
            paired = self.torch.einsum(
                "qrh,crh->cqh",
                local_kernel,
                paired_source[:, :, :width],
            )
            spectrum[:4, q_local, :width] = paired[:4]
            if width > 1:
                # For exp(i*x*cos(theta)), K_{-h}=K_h.  Only the FFT of the
                # real source histogram is conjugated at negative h;
                # conjugating the already multiplied positive coefficient
                # would incorrectly conjugate the complex Bessel phase i**h.
                spectrum[:4, q_local, -(width - 1) :] = self.torch.flip(
                    paired[4:, :, 1:], dims=(-1,)
                )
        values = self.torch.fft.ifft(spectrum, dim=-1)
        if self.phase_sign == -1:
            values = self.torch.conj(values)
        return TorchMagneticSansAmplitudes(
            nuclear=values[0],
            magnetization=values[1:].permute(1, 2, 0).contiguous(),
        )

    def amplitudes(self) -> TorchMagneticSansAmplitudes:
        """Evaluate all four prepared Fourier channels without leaving the device."""

        return self._amplitudes_for_source(self.source_crh)

    def amplitudes_from_weights(self, channel_weights: Any) -> TorchMagneticSansAmplitudes:
        """Evaluate a new field state, including histogram and angular FFT."""

        return self._amplitudes_for_source(self.source_spectrum_from_weights(channel_weights))

    def _cross_sections_from_amplitudes(
        self, amplitudes: TorchMagneticSansAmplitudes, polarization: Any
    ) -> dict[str, Any]:
        p = self.torch.as_tensor(
            polarization,
            dtype=self.real_dtype,
            device=self.device,
        )
        p = p / self.torch.linalg.vector_norm(p)
        q_hat = self.q_hat_phi3.unsqueeze(0)
        magnetization = amplitudes.magnetization
        interaction = q_hat * self.torch.sum(q_hat * magnetization, dim=-1, keepdim=True)
        interaction = interaction - magnetization
        p_complex = p.to(self.complex_dtype)
        projected_amp = self.torch.einsum("qpi,i->qp", interaction, p_complex)
        nuclear = self.torch.abs(amplitudes.nuclear) ** 2
        magnetic = self.torch.sum(self.torch.abs(interaction) ** 2, dim=-1)
        projected = self.torch.abs(projected_amp) ** 2
        nuclear_magnetic = 2.0 * self.torch.real(
            amplitudes.nuclear * self.torch.conj(projected_amp)
        )
        spin_flip = magnetic - projected
        chiral = self.torch.real(
            -1j
            * self.torch.einsum(
                "i,qpi->qp",
                p_complex,
                self.torch.linalg.cross(interaction, self.torch.conj(interaction)),
            )
        )
        plus_minus = spin_flip + chiral
        minus_plus = spin_flip - chiral
        plus_plus = nuclear + nuclear_magnetic + projected
        minus_minus = nuclear - nuclear_magnetic + projected
        return {
            "S_N": nuclear,
            "S_M": magnetic,
            "S_NM": nuclear_magnetic,
            "S_P": projected,
            "S_chi": chiral,
            "S_sf": spin_flip,
            "S_pm": plus_minus,
            "S_mp": minus_plus,
            "S_pp": plus_plus,
            "S_mm": minus_minus,
            "S_p": plus_plus + plus_minus,
            "S_m": minus_minus + minus_plus,
        }

    def cross_sections(self, polarization: Any) -> dict[str, Any]:
        """Return twelve observables for the prepared field state."""

        return self._cross_sections_from_amplitudes(self.amplitudes(), polarization)

    def cross_sections_from_weights(self, channel_weights: Any, polarization: Any) -> dict[str, Any]:
        """Return twelve observables for a new GPU-resident field state."""

        return self._cross_sections_from_amplitudes(
            self.amplitudes_from_weights(channel_weights), polarization
        )
