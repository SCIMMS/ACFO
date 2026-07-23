"""Independent scalar anisotropic-Born reference for ACFO validation.

The reference path samples a contrast field on a Cartesian midpoint grid,
solves the first-Born spectral source problem with a zero-padded 3-D FFT, and
interpolates the resulting Cartesian spectrum onto PDE-derived dispersion
branches. It intentionally does not use cylindrical harmonics or ACFO kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .axisymmetric_manifold import AxisymmetricManifold

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


Branch = Literal["ordinary", "extraordinary"]


@dataclass(frozen=True)
class UniaxialScalarDispersion:
    """Scalar ordinary and extraordinary branches of a uniaxial medium."""

    epsilon_parallel: float
    epsilon_perpendicular: float
    k0: float

    def __post_init__(self) -> None:
        for name, value in (
            ("epsilon_parallel", self.epsilon_parallel),
            ("epsilon_perpendicular", self.epsilon_perpendicular),
            ("k0", self.k0),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def incident_kz(self) -> float:
        """Incident wave number for propagation along the optic axis."""

        return float(np.sqrt(self.epsilon_perpendicular) * self.k0)

    def manifold(self, u: "ArrayLike", branch: Branch) -> AxisymmetricManifold:
        """Return scattering-vector nodes ``k_out - k_incident``."""

        u_array = np.asarray(u, dtype=np.float64)
        if u_array.ndim != 1 or u_array.size == 0 or not np.all(np.isfinite(u_array)):
            raise ValueError("u must be a non-empty finite vector")
        if np.any(np.diff(u_array) <= 0.0):
            raise ValueError("u must be strictly increasing")
        if np.any((u_array < 0.0) | (u_array >= 0.5 * np.pi)):
            raise ValueError("u must lie in [0, pi/2) for the outgoing branch")
        if branch == "ordinary":
            transverse_scale = np.sqrt(self.epsilon_perpendicular) * self.k0
        elif branch == "extraordinary":
            transverse_scale = np.sqrt(self.epsilon_parallel) * self.k0
        else:
            raise ValueError("branch must be ordinary or extraordinary")
        q_perp = transverse_scale * np.sin(u_array)
        q_z = self.incident_kz * (np.cos(u_array) - 1.0)
        return AxisymmetricManifold(
            u_array,
            q_perp,
            q_z,
            name=f"uniaxial-{branch}-branch",
            interpretation="dispersion-derived",
        )

    def outgoing_residue_weight(
        self,
        u: "ArrayLike",
        branch: Branch,
    ) -> "NDArray[np.float64]":
        """Return the scalar pole-residue weight, excluding a global constant."""

        u_array = np.asarray(u, dtype=np.float64)
        cosine = np.cos(u_array)
        if u_array.ndim != 1 or np.any(cosine <= 0.0):
            raise ValueError("u must be a vector below the grazing angle")
        if branch == "ordinary":
            weight = 1.0 / (2.0 * self.incident_kz * cosine)
        elif branch == "extraordinary":
            weight = self.epsilon_perpendicular / (
                2.0 * self.incident_kz * cosine
            )
        else:
            raise ValueError("branch must be ordinary or extraordinary")
        return weight.astype(np.float64, copy=False)


class CartesianSpectralBornReference:
    """Zero-padded Cartesian FFT reference for positive-sign Fourier samples."""

    def __init__(
        self,
        density: "ArrayLike",
        *,
        half_width: float,
        padding_factor: int = 4,
    ) -> None:
        values = np.asarray(density, dtype=np.complex128)
        if values.ndim != 3 or len(set(values.shape)) != 1 or values.shape[0] < 2:
            raise ValueError("density must be a nontrivial cubic 3-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("density must contain only finite values")
        half_width = float(half_width)
        padding_factor = int(padding_factor)
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError("half_width must be finite and positive")
        if padding_factor < 1:
            raise ValueError("padding_factor must be positive")

        n = values.shape[0]
        padded_n = padding_factor * n
        spacing = 2.0 * half_width / n
        padded = np.zeros((padded_n, padded_n, padded_n), dtype=np.complex128)
        padded[:n, :n, :n] = values
        spectrum = np.fft.ifftn(padded) * (padded_n**3 * spacing**3)
        frequencies = np.fft.fftshift(
            2.0 * np.pi * np.fft.fftfreq(padded_n, d=spacing)
        )
        spectrum = np.fft.fftshift(spectrum)
        first_center = -half_width + 0.5 * spacing
        one_dimensional_phase = np.exp(1j * first_center * frequencies)
        spectrum *= one_dimensional_phase[:, None, None]
        spectrum *= one_dimensional_phase[None, :, None]
        spectrum *= one_dimensional_phase[None, None, :]

        self.n_per_axis = n
        self.padded_n = padded_n
        self.half_width = half_width
        self.spacing = spacing
        self.frequencies = frequencies
        self._interpolator = RegularGridInterpolator(
            (frequencies, frequencies, frequencies),
            spectrum,
            method="linear",
            bounds_error=True,
        )

    def fourier_nodes(self, q_nodes: "ArrayLike") -> "NDArray[np.complex128]":
        """Interpolate the Cartesian spectral solution at arbitrary nodes."""

        nodes = np.asarray(q_nodes, dtype=np.float64)
        if nodes.ndim < 2 or nodes.shape[-1] != 3 or not np.all(np.isfinite(nodes)):
            raise ValueError("q_nodes must be finite and end in a Cartesian axis")
        result = self._interpolator(nodes.reshape(-1, 3))
        return np.asarray(result, dtype=np.complex128).reshape(nodes.shape[:-1])

    def born_field(
        self,
        manifold: AxisymmetricManifold,
        phi: "ArrayLike",
        radial_weight: "ArrayLike | None" = None,
    ) -> "NDArray[np.complex128]":
        """Return the scalar first-Born outgoing modal field."""

        amplitude = self.fourier_nodes(manifold.target_nodes(phi))
        if radial_weight is None:
            return amplitude
        weight = np.asarray(radial_weight, dtype=np.float64)
        if weight.shape != (manifold.n_u,) or not np.all(np.isfinite(weight)):
            raise ValueError(f"radial_weight must be finite with shape ({manifold.n_u},)")
        return amplitude * weight[:, None]
