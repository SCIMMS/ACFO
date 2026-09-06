"""Polarized magnetic SANS observables on prepared cylindrical q rings.

The Fourier stage evaluates one scalar nuclear channel and three Cartesian
magnetization channels.  The inexpensive detector-local contraction then
forms the Halpern--Johnson interaction vector and the four POLARIS spin
cross-sections.  Overall dimensional prefactors are explicit so that the
operator can be validated independently of a particular unit convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .axisymmetric_manifold import AxisymmetricManifold
from .axisymmetric_operator import PreparedAxisymmetricOperator
from .histogram import make_cylindrical_histogram
from .modal_coupling import PreparedAngularModeCoupling
from .solvers import PreparedCakePlan


ArrayLike = Any


def _halpern_johnson_modal_coupling(
    q_perp: np.ndarray,
    q_z: np.ndarray,
    phi_origin: float,
    n_phi: int,
) -> PreparedAngularModeCoupling:
    """Prepare the exact five-mode Halpern--Johnson projector analytically."""

    radial = np.asarray(q_perp, dtype=np.float64)
    axial = np.asarray(q_z, dtype=np.float64)
    if radial.ndim != 1 or axial.shape != radial.shape:
        raise ValueError("q_perp and q_z must be matching one-dimensional arrays")
    q_norm = np.hypot(radial, axial)
    if np.any(q_norm <= 0.0):
        raise ValueError("Halpern--Johnson projection is undefined at q=0")
    a = radial / q_norm
    b = axial / q_norm
    a2 = a * a
    ab = a * b
    shifts = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    coefficients = np.zeros((radial.size, 5, 3, 3), dtype=np.complex128)

    zero = coefficients[:, 2]
    zero[:, 0, 0] = 0.5 * a2 - 1.0
    zero[:, 1, 1] = 0.5 * a2 - 1.0
    zero[:, 2, 2] = b * b - 1.0

    plus_one = coefficients[:, 3]
    plus_one[:, 0, 2] = plus_one[:, 2, 0] = 0.5 * ab
    plus_one[:, 1, 2] = plus_one[:, 2, 1] = -0.5j * ab
    coefficients[:, 1] = np.conj(plus_one)

    plus_two = coefficients[:, 4]
    plus_two[:, 0, 0] = 0.25 * a2
    plus_two[:, 1, 1] = -0.25 * a2
    plus_two[:, 0, 1] = plus_two[:, 1, 0] = -0.25j * a2
    coefficients[:, 0] = np.conj(plus_two)

    # The DFT is taken over detector indices, while the physical samples are
    # phi_origin + 2*pi*k/n_phi.  Fold that origin into the modal coefficients.
    coefficients *= np.exp(1j * shifts * float(phi_origin))[None, :, None, None]
    return PreparedAngularModeCoupling.from_mode_coefficients(
        coefficients,
        shifts,
        n_phi=int(n_phi),
    )


def load_numagsans_magnetization_csv(
    path: str | Path,
    *,
    rotate_beam_x_to_z: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a public NuMagSANS ``x y z Mx My Mz`` magnetization file.

    NuMagSANS uses the incident beam along ``+x`` and the detector in the
    ``yz`` plane.  ACFO's cylindrical operators use ``+z`` as the orbit axis.
    The optional proper coordinate permutation maps
    ``(x, y, z)_NuMagSANS -> (y, z, x)_ACFO`` for both positions and vectors.
    """

    source = Path(path)
    values = np.loadtxt(source, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("NuMagSANS magnetization CSV must contain at least six columns")
    if not np.all(np.isfinite(values[:, :6])):
        raise ValueError("NuMagSANS magnetization CSV contains non-finite values")
    points = values[:, :3]
    magnetization = values[:, 3:6]
    if rotate_beam_x_to_z:
        permutation = np.asarray([1, 2, 0], dtype=np.intp)
        points = points[:, permutation]
        magnetization = magnetization[:, permutation]
    return np.ascontiguousarray(points), np.ascontiguousarray(magnetization)


def load_numagsans_nuclear_csv(
    path: str | Path,
    *,
    rotate_beam_x_to_z: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a public NuMagSANS ``x y z nuclear_weight`` input file."""

    source = Path(path)
    values = np.loadtxt(source, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("NuMagSANS nuclear CSV must contain at least four columns")
    if not np.all(np.isfinite(values[:, :4])):
        raise ValueError("NuMagSANS nuclear CSV contains non-finite values")
    points = values[:, :3]
    if rotate_beam_x_to_z:
        points = points[:, np.asarray([1, 2, 0], dtype=np.intp)]
    return np.ascontiguousarray(points), np.ascontiguousarray(values[:, 3])


def load_numagsans_fourier_sources(
    magnetization_path: str | Path,
    nuclear_path: str | Path,
    *,
    rotate_beam_x_to_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and coordinate-match NuMagSANS nuclear/magnetization source files."""

    magnetic_points, magnetization = load_numagsans_magnetization_csv(
        magnetization_path,
        rotate_beam_x_to_z=rotate_beam_x_to_z,
    )
    nuclear_points, nuclear = load_numagsans_nuclear_csv(
        nuclear_path,
        rotate_beam_x_to_z=rotate_beam_x_to_z,
    )
    if magnetic_points.shape != nuclear_points.shape or not np.array_equal(
        magnetic_points,
        nuclear_points,
    ):
        raise ValueError("NuMagSANS nuclear and magnetization coordinates do not match")
    return magnetic_points, nuclear, magnetization


def _q_array(q_xyz: ArrayLike) -> np.ndarray:
    q = np.asarray(q_xyz, dtype=np.float64)
    if q.ndim < 2 or q.shape[-1] != 3:
        raise ValueError("q_xyz must have shape (..., 3)")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_xyz must contain only finite values")
    return q


def _channel_array(values: ArrayLike, target_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != target_shape:
        raise ValueError(f"{name} must have shape {target_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class MagneticSansAmplitudes:
    """Nuclear and vector-magnetization Fourier amplitudes."""

    nuclear: np.ndarray
    magnetization: np.ndarray
    q_xyz: np.ndarray


@dataclass(frozen=True)
class PolarizedSansCrossSections:
    """Four POLARIS channels and their reusable physical building blocks."""

    plus_plus: np.ndarray
    minus_minus: np.ndarray
    plus_minus: np.ndarray
    minus_plus: np.ndarray
    nuclear: np.ndarray
    magnetic: np.ndarray
    projected: np.ndarray
    nuclear_magnetic: np.ndarray
    spin_flip: np.ndarray
    chiral: np.ndarray
    interaction_vector: np.ndarray


def halpern_johnson(
    magnetization_fourier: ArrayLike,
    q_xyz: ArrayLike,
) -> np.ndarray:
    """Return ``qhat x (qhat x M_tilde)`` on arbitrary nonzero q nodes."""

    q = _q_array(q_xyz)
    magnetization = np.asarray(magnetization_fourier, dtype=np.complex128)
    if magnetization.shape != q.shape:
        raise ValueError("magnetization_fourier and q_xyz must have matching shapes")
    q_norm = np.linalg.norm(q, axis=-1)
    if np.any(q_norm <= 0.0):
        raise ValueError("Halpern--Johnson projection is undefined at q=0")
    q_hat = q / q_norm[..., None]
    return np.cross(q_hat, np.cross(q_hat, magnetization))


def polarized_sans_cross_sections(
    nuclear_fourier: ArrayLike,
    magnetization_fourier: ArrayLike,
    q_xyz: ArrayLike,
    polarization: ArrayLike,
    *,
    magnetic_scattering_length: float = 1.0,
    prefactor: float = 1.0,
) -> PolarizedSansCrossSections:
    """Contract Fourier channels into invariant polarized-SANS observables.

    The returned convention follows

    ``sf = magnetic - projected`` and
    ``chiral = -i b_H^2 P . (Q x Q*)``.

    Consequently ``plus_minus = sf + chiral`` and
    ``minus_plus = sf - chiral``.  ``prefactor`` may be set to ``8*pi^3/V``
    when absolute cross-sections are required.
    """

    q = _q_array(q_xyz)
    target_shape = q.shape[:-1]
    nuclear_amp = _channel_array(nuclear_fourier, target_shape, "nuclear_fourier")
    magnetization = np.asarray(magnetization_fourier, dtype=np.complex128)
    if magnetization.shape != q.shape:
        raise ValueError("magnetization_fourier must have shape matching q_xyz")

    p = np.asarray(polarization, dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("polarization must be a finite three-vector")
    p_norm = float(np.linalg.norm(p))
    if p_norm <= 0.0:
        raise ValueError("polarization must be nonzero")
    p = p / p_norm

    b_h = float(magnetic_scattering_length)
    scale = float(prefactor)
    if not np.isfinite(b_h) or not np.isfinite(scale) or scale < 0.0:
        raise ValueError("magnetic_scattering_length and prefactor must be finite")

    interaction = halpern_johnson(magnetization, q)
    projected_amp = np.einsum("...i,i->...", interaction, p)
    nuclear = np.abs(nuclear_amp) ** 2
    magnetic = b_h * b_h * np.sum(np.abs(interaction) ** 2, axis=-1)
    projected = b_h * b_h * np.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * b_h * np.real(nuclear_amp * np.conj(projected_amp))
    spin_flip = magnetic - projected
    chiral_complex = -1j * b_h * b_h * np.einsum(
        "i,...i->...",
        p,
        np.cross(interaction, np.conj(interaction)),
    )
    chiral = np.real(chiral_complex)

    plus_plus = nuclear + nuclear_magnetic + projected
    minus_minus = nuclear - nuclear_magnetic + projected
    plus_minus = spin_flip + chiral
    minus_plus = spin_flip - chiral

    def scaled_real(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(scale * np.real(value), dtype=np.float64)

    return PolarizedSansCrossSections(
        plus_plus=scaled_real(plus_plus),
        minus_minus=scaled_real(minus_minus),
        plus_minus=scaled_real(plus_minus),
        minus_plus=scaled_real(minus_plus),
        nuclear=scaled_real(nuclear),
        magnetic=scaled_real(magnetic),
        projected=scaled_real(projected),
        nuclear_magnetic=scaled_real(nuclear_magnetic),
        spin_flip=scaled_real(spin_flip),
        chiral=scaled_real(chiral),
        interaction_vector=np.ascontiguousarray(interaction),
    )


def polarized_sans_cross_sections_from_interaction(
    nuclear_fourier: ArrayLike,
    interaction_vector: ArrayLike,
    polarization: ArrayLike,
    *,
    magnetic_scattering_length: float = 1.0,
    prefactor: float = 1.0,
) -> PolarizedSansCrossSections:
    """Form polarized-SANS observables from an already projected amplitude.

    This entry point avoids recomputing the pointwise Halpern--Johnson
    projector when it has already been applied as a finite-band angular-mode
    coupling.
    """

    interaction = np.asarray(interaction_vector, dtype=np.complex128)
    if interaction.ndim < 2 or interaction.shape[-1] != 3:
        raise ValueError("interaction_vector must have shape (..., 3)")
    if not np.all(np.isfinite(interaction)):
        raise ValueError("interaction_vector must contain only finite values")
    target_shape = interaction.shape[:-1]
    nuclear_amp = _channel_array(nuclear_fourier, target_shape, "nuclear_fourier")
    p = np.asarray(polarization, dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("polarization must be a finite three-vector")
    p_norm = float(np.linalg.norm(p))
    if p_norm <= 0.0:
        raise ValueError("polarization must be nonzero")
    p = p / p_norm

    b_h = float(magnetic_scattering_length)
    scale = float(prefactor)
    if not np.isfinite(b_h) or not np.isfinite(scale) or scale < 0.0:
        raise ValueError("magnetic_scattering_length and prefactor must be finite")

    projected_amp = np.einsum("...i,i->...", interaction, p)
    nuclear = np.abs(nuclear_amp) ** 2
    magnetic = b_h * b_h * np.sum(np.abs(interaction) ** 2, axis=-1)
    projected = b_h * b_h * np.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * b_h * np.real(nuclear_amp * np.conj(projected_amp))
    spin_flip = magnetic - projected
    chiral = np.real(
        -1j
        * b_h
        * b_h
        * np.einsum("i,...i->...", p, np.cross(interaction, np.conj(interaction)))
    )

    def scaled_real(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(scale * np.real(value), dtype=np.float64)

    return PolarizedSansCrossSections(
        plus_plus=scaled_real(nuclear + nuclear_magnetic + projected),
        minus_minus=scaled_real(nuclear - nuclear_magnetic + projected),
        plus_minus=scaled_real(spin_flip + chiral),
        minus_plus=scaled_real(spin_flip - chiral),
        nuclear=scaled_real(nuclear),
        magnetic=scaled_real(magnetic),
        projected=scaled_real(projected),
        nuclear_magnetic=scaled_real(nuclear_magnetic),
        spin_flip=scaled_real(spin_flip),
        chiral=scaled_real(chiral),
        interaction_vector=np.ascontiguousarray(interaction),
    )


def direct_fourier_channels(
    points: ArrayLike,
    nuclear_weights: ArrayLike,
    magnetization_weights: ArrayLike,
    q_xyz: ArrayLike,
    *,
    phase_sign: int = -1,
    q_block_size: int = 256,
) -> MagneticSansAmplitudes:
    """Direct complex exponent-sum oracle for four SANS source channels."""

    coords = np.asarray(points, dtype=np.float64)
    nuclear = np.asarray(nuclear_weights, dtype=np.complex128)
    magnetization = np.asarray(magnetization_weights, dtype=np.complex128)
    q = _q_array(q_xyz)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if nuclear.shape != (coords.shape[0],):
        raise ValueError("nuclear_weights must have shape (N,)")
    if magnetization.shape != coords.shape:
        raise ValueError("magnetization_weights must have shape (N, 3)")
    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    block_size = int(q_block_size)
    if block_size <= 0:
        raise ValueError("q_block_size must be positive")

    flat_q = q.reshape(-1, 3)
    nuclear_out = np.empty(flat_q.shape[0], dtype=np.complex128)
    magnetic_out = np.empty((flat_q.shape[0], 3), dtype=np.complex128)
    channels = np.column_stack([nuclear, magnetization])
    for start in range(0, flat_q.shape[0], block_size):
        local = slice(start, min(start + block_size, flat_q.shape[0]))
        phase = np.exp(1j * float(phase_sign) * (flat_q[local] @ coords.T))
        transformed = phase @ channels
        nuclear_out[local] = transformed[:, 0]
        magnetic_out[local] = transformed[:, 1:]
    return MagneticSansAmplitudes(
        nuclear=nuclear_out.reshape(q.shape[:-1]),
        magnetization=magnetic_out.reshape(q.shape),
        q_xyz=np.array(q, copy=True),
    )


class PreparedMagneticSansOperator:
    """Prepared four-channel ACFO evaluator followed by SANS contractions.

    ACFO internally uses the positive Fourier phase.  For the conventional
    SANS negative phase, the identity
    ``A_-(w) = conj(A_+(conj(w)))`` is applied channel by channel.
    """

    def __init__(
        self,
        points: ArrayLike,
        nuclear_weights: ArrayLike,
        magnetization_weights: ArrayLike,
        manifold: AxisymmetricManifold,
        *,
        n_r: int,
        n_z: int,
        n_phi: int,
        r_max: float | None = None,
        z_range: tuple[float, float] | None = None,
        phase_sign: int = -1,
        hist_backend: str = "numpy",
        complex_dtype: np.dtype | str = np.complex128,
    ) -> None:
        coords = np.asarray(points, dtype=np.float64)
        nuclear = np.asarray(nuclear_weights)
        magnetization = np.asarray(magnetization_weights)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if nuclear.shape != (coords.shape[0],):
            raise ValueError("nuclear_weights must have shape (N,)")
        if magnetization.shape != coords.shape:
            raise ValueError("magnetization_weights must have shape (N, 3)")
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1")

        # Retain the fixed source coordinates for exact q_z=0 compaction in
        # the GPU wrapper.  The hot magnetic-SANS path can then determine the
        # occupied cylindrical radii from geometry rather than from the
        # initial field values, which remains correct when later field updates
        # turn an initially zero weight into a nonzero one.
        self.source_points = np.ascontiguousarray(coords, dtype=np.float64)

        source_channels = np.column_stack([nuclear, magnetization]).astype(
            np.complex128,
            copy=False,
        )
        if phase_sign == -1:
            source_channels = np.conj(source_channels)

        binned = tuple(
            make_cylindrical_histogram(
                coords,
                atom_weights=source_channels[:, channel],
                n_r=int(n_r),
                n_z=int(n_z),
                n_phi=int(n_phi),
                r_max=r_max,
                z_range=z_range,
                backend=hist_backend,
                hist_dtype=np.complex128,
            )
            for channel in range(4)
        )
        template = binned[0]
        self.manifold = manifold
        self.phase_sign = int(phase_sign)
        self.phi = np.array(template.beta_centers, copy=True)
        self.operator = PreparedAxisymmetricOperator(
            template,
            manifold,
            complex_dtype=complex_dtype,
        )
        self.channel_fourier = np.ascontiguousarray(
            np.stack(
                [
                    self.operator.prepare_object(
                        np.asarray(item.hist, dtype=self.operator.complex_dtype)
                    )
                    for item in binned
                ],
                axis=0,
            )
        )
        self.modal_interaction = _halpern_johnson_modal_coupling(
            manifold.q_perp,
            manifold.q_z,
            self.phi[0],
            self.phi.size,
        )

    @property
    def q_xyz(self) -> np.ndarray:
        qx = self.manifold.q_perp[:, None] * np.cos(self.phi)[None, :]
        qy = self.manifold.q_perp[:, None] * np.sin(self.phi)[None, :]
        qz = np.broadcast_to(self.manifold.q_z[:, None], qx.shape)
        return np.stack([qx, qy, qz], axis=-1)

    def amplitude_spectra(self) -> np.ndarray:
        """Return four standard DFT spectra before the detector-angle IFFT."""

        transformed = np.stack(
            [
                self.operator.apply_prepared_object_fourier(values)
                for values in self.channel_fourier
            ],
            axis=0,
        )
        if self.phase_sign == -1:
            reverse = (-np.arange(self.phi.size, dtype=np.int64)) % self.phi.size
            transformed = np.conj(transformed[..., reverse])
        return np.ascontiguousarray(transformed)

    def amplitudes(self) -> MagneticSansAmplitudes:
        transformed = np.fft.ifft(self.amplitude_spectra(), axis=-1)
        return MagneticSansAmplitudes(
            nuclear=np.ascontiguousarray(transformed[0]),
            magnetization=np.ascontiguousarray(np.moveaxis(transformed[1:], 0, -1)),
            q_xyz=self.q_xyz,
        )

    def cross_sections(
        self,
        polarization: ArrayLike,
        *,
        magnetic_scattering_length: float = 1.0,
        prefactor: float = 1.0,
    ) -> PolarizedSansCrossSections:
        amplitudes = self.amplitudes()
        return polarized_sans_cross_sections(
            amplitudes.nuclear,
            amplitudes.magnetization,
            amplitudes.q_xyz,
            polarization,
            magnetic_scattering_length=magnetic_scattering_length,
            prefactor=prefactor,
        )

    def cross_sections_modal(
        self,
        polarization: ArrayLike,
        *,
        magnetic_scattering_length: float = 1.0,
        prefactor: float = 1.0,
    ) -> PolarizedSansCrossSections:
        """Evaluate the Halpern--Johnson projector before angular synthesis."""

        spectra = self.amplitude_spectra()
        nuclear = np.fft.ifft(spectra[0], axis=-1)
        interaction = self.modal_interaction.synthesize_spectrum(spectra[1:4])
        return polarized_sans_cross_sections_from_interaction(
            nuclear,
            interaction,
            polarization,
            magnetic_scattering_length=magnetic_scattering_length,
            prefactor=prefactor,
        )


class PreparedBandlimitedMagneticSansOperator:
    """Four-channel magnetic-SANS wrapper using the production cake solver.

    Unlike :class:`PreparedMagneticSansOperator`, this path applies an
    R-dependent angular cutoff.  It is intended for large ``n_phi`` detectors
    where the physical support ``q_perp * R`` occupies only a small fraction
    of the available Fourier modes.  The four channels currently share the
    geometry contract but are evaluated by four component plans; a fused
    multi-channel contraction remains future work.
    """

    def __init__(
        self,
        points: ArrayLike,
        nuclear_weights: ArrayLike,
        magnetization_weights: ArrayLike,
        manifold: AxisymmetricManifold,
        *,
        n_r: int,
        n_z: int,
        n_phi: int,
        r_max: float | None = None,
        z_range: tuple[float, float] | None = None,
        phase_sign: int = -1,
        hist_backend: str = "numpy",
        circular_backend: str = "auto",
        complex_dtype: np.dtype | str = np.complex128,
        margin: int = 16,
        cutoff_bin_size: int = 16,
        analytic_kernel: bool = True,
        fused_analytic_kernel: bool = True,
        q_block_size: int = 128,
        prepare_modal_interaction: bool = True,
    ) -> None:
        coords = np.asarray(points, dtype=np.float64)
        nuclear = np.asarray(nuclear_weights)
        magnetization = np.asarray(magnetization_weights)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if nuclear.shape != (coords.shape[0],):
            raise ValueError("nuclear_weights must have shape (N,)")
        if magnetization.shape != coords.shape:
            raise ValueError("magnetization_weights must have shape (N, 3)")
        if phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or +1")
        if margin < 0 or cutoff_bin_size <= 0:
            raise ValueError("margin must be non-negative and cutoff_bin_size positive")

        self.source_points = np.ascontiguousarray(coords, dtype=np.float64)

        source_channels = np.column_stack([nuclear, magnetization]).astype(
            np.complex128,
            copy=False,
        )
        if phase_sign == -1:
            source_channels = np.conj(source_channels)
        use_real_histograms = not bool(np.any(source_channels.imag))
        histogram_dtype = np.float64 if use_real_histograms else np.complex128

        binned = tuple(
            make_cylindrical_histogram(
                coords,
                atom_weights=(
                    source_channels[:, channel].real
                    if use_real_histograms
                    else source_channels[:, channel]
                ),
                n_r=int(n_r),
                n_z=int(n_z),
                n_phi=int(n_phi),
                r_max=r_max,
                z_range=z_range,
                backend=hist_backend,
                hist_dtype=histogram_dtype,
            )
            for channel in range(4)
        )
        self.manifold = manifold
        self.phase_sign = int(phase_sign)
        self.phi = np.array(binned[0].beta_centers, copy=True)
        self.margin = int(margin)
        self.cutoff_bin_size = int(cutoff_bin_size)
        self.analytic_kernel = bool(analytic_kernel)
        self.fused_analytic_kernel = bool(fused_analytic_kernel)
        self.q_block_size = int(q_block_size)
        self.channel_plans = tuple(
            PreparedCakePlan(
                item,
                manifold.q_norm,
                1.0,
                phi=self.phi,
                circular_backend=circular_backend,
                complex_dtype=complex_dtype,
                q_perp=manifold.q_perp,
                q_z=manifold.q_z,
                q_block_size=self.q_block_size,
            )
            for item in binned
        )
        self.modal_interaction = (
            _halpern_johnson_modal_coupling(
                manifold.q_perp,
                manifold.q_z,
                self.phi[0],
                self.phi.size,
            )
            if prepare_modal_interaction
            else None
        )
        max_source_mode = int(
            np.ceil(
                np.max(np.abs(manifold.q_perp))
                * np.max(self.channel_plans[0].binned.r_centers)
                + self.margin
            )
        )
        max_source_mode = min(max_source_mode, self.phi.size // 2)
        max_source_mode = min(
            (
                (max_source_mode + self.cutoff_bin_size - 1)
                // self.cutoff_bin_size
            )
            * self.cutoff_bin_size,
            self.phi.size // 2,
        )
        if 2 * max_source_mode + 1 < self.phi.size:
            self.modal_modes: np.ndarray | None = np.arange(
                -max_source_mode,
                max_source_mode + 1,
                dtype=np.int64,
            )
            self.modal_modes.setflags(write=False)
        else:
            self.modal_modes = None

    @property
    def q_xyz(self) -> np.ndarray:
        qx = self.manifold.q_perp[:, None] * np.cos(self.phi)[None, :]
        qy = self.manifold.q_perp[:, None] * np.sin(self.phi)[None, :]
        qz = np.broadcast_to(self.manifold.q_z[:, None], qx.shape)
        return np.stack([qx, qy, qz], axis=-1)

    def amplitude_spectra(self) -> np.ndarray:
        """Return four bandlimited standard DFT spectra before synthesis."""

        transformed = np.stack(
            [
                plan.circular_ahat_r_dependent_bandlimit(
                    margin=self.margin,
                    cutoff_bin_size=self.cutoff_bin_size,
                    analytic_kernel=self.analytic_kernel,
                    fused_analytic_kernel=self.fused_analytic_kernel,
                    q_block_size=self.q_block_size,
                )
                for plan in self.channel_plans
            ],
            axis=0,
        )
        if self.phase_sign == -1:
            reverse = (-np.arange(self.phi.size, dtype=np.int64)) % self.phi.size
            transformed = np.conj(transformed[..., reverse])
        return np.ascontiguousarray(transformed)

    def amplitudes(self) -> MagneticSansAmplitudes:
        transformed = np.fft.ifft(self.amplitude_spectra(), axis=-1)
        return MagneticSansAmplitudes(
            nuclear=np.ascontiguousarray(transformed[0]),
            magnetization=np.ascontiguousarray(np.moveaxis(transformed[1:], 0, -1)),
            q_xyz=self.q_xyz,
        )

    def cross_sections(
        self,
        polarization: ArrayLike,
        *,
        magnetic_scattering_length: float = 1.0,
        prefactor: float = 1.0,
    ) -> PolarizedSansCrossSections:
        amplitudes = self.amplitudes()
        return polarized_sans_cross_sections(
            amplitudes.nuclear,
            amplitudes.magnetization,
            amplitudes.q_xyz,
            polarization,
            magnetic_scattering_length=magnetic_scattering_length,
            prefactor=prefactor,
        )

    def cross_sections_modal(
        self,
        polarization: ArrayLike,
        *,
        magnetic_scattering_length: float = 1.0,
        prefactor: float = 1.0,
    ) -> PolarizedSansCrossSections:
        """Project bandlimited magnetic modes before angular synthesis."""

        spectra = self.amplitude_spectra()
        nuclear = np.fft.ifft(spectra[0], axis=-1)
        if self.modal_modes is None:
            interaction = self.modal_interaction.synthesize_spectrum(spectra[1:4])
        else:
            interaction = self.modal_interaction.synthesize_compact_spectrum(
                spectra[1:4, :, self.modal_modes % self.phi.size],
                self.modal_modes,
            )
        return polarized_sans_cross_sections_from_interaction(
            nuclear,
            interaction,
            polarization,
            magnetic_scattering_length=magnetic_scattering_length,
            prefactor=prefactor,
        )
