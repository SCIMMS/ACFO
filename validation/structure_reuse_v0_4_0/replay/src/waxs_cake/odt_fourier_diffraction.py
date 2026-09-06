"""Physical Fourier-diffraction calibration for weak-scattering ODT.

The prepared ACFO operator evaluates the unnormalised positive-sign Fourier
integral of an object represented by integrated cylindrical coefficients.
ODTbrain forms a negative-sign two-dimensional FFT of a Born or Rytov field.
With the symmetric Fourier convention used in the ODTbrain derivation,
conversion to the corresponding unnormalised object transform reduces to

    F_raw(q) = -2 i k_z U_raw(k_x, k_y).

Here ``U_raw`` includes detector-pixel area.  Reinterpreting the result as the
positive-sign transform of the spatially reflected object preserves real and
non-negative object constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def rytov_fft_to_object_fourier_samples(
    spectrum: "ArrayLike",
    base_q: "ArrayLike",
    *,
    medium_wavenumber: float,
    detector_pixel_size: "ArrayLike | float",
    detector_normal: "ArrayLike" = (0.0, 0.0, 1.0),
) -> "NDArray[np.complex128]":
    """Convert a centred NumPy Rytov FFT to physical object-FT samples.

    ``spectrum`` may have any leading dimensions, but its last dimension must
    match ``base_q``.  ``base_q`` is the unrotated Ewald scattering vector.
    The longitudinal outgoing wavenumber is
    ``k_z = k_m + q dot detector_normal``.

    The ODTbrain Fourier diffraction theorem is

    ``F_hat = -sqrt(2/pi) i k_m M U_hat``.

    Converting its symmetric 2-D/3-D Fourier normalisations to raw integrals
    gives ``F_raw = -2 i k_z U_raw``.  A sampled detector integral contributes
    the positive factor ``dx*dy``.
    """

    values = np.asarray(spectrum, dtype=np.complex128)
    q = np.asarray(base_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or q.shape[0] == 0:
        raise ValueError("base_q must have shape (n_orbit, 3)")
    if values.ndim == 0 or values.shape[-1] != q.shape[0]:
        raise ValueError("the final spectrum dimension must match base_q")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(q)):
        raise ValueError("spectrum and base_q must contain only finite values")

    k_medium = float(medium_wavenumber)
    if not np.isfinite(k_medium) or k_medium <= 0.0:
        raise ValueError("medium_wavenumber must be finite and positive")

    pixel = np.asarray(detector_pixel_size, dtype=np.float64)
    if pixel.ndim == 0:
        pixel = np.repeat(pixel.reshape(1), 2)
    if (
        pixel.shape != (2,)
        or not np.all(np.isfinite(pixel))
        or np.any(pixel <= 0.0)
    ):
        raise ValueError(
            "detector_pixel_size must be a positive scalar or two-vector"
        )

    normal = np.asarray(detector_normal, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("detector_normal must be a finite three-vector")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0.0:
        raise ValueError("detector_normal must be nonzero")
    normal = normal / normal_norm

    outgoing_longitudinal = k_medium + q @ normal
    tolerance = 64.0 * np.finfo(np.float64).eps * k_medium
    if np.any(outgoing_longitudinal < -tolerance):
        raise ValueError("base_q contains non-propagating longitudinal samples")
    outgoing_longitudinal = np.maximum(outgoing_longitudinal, 0.0)
    pixel_area = float(pixel[0] * pixel[1])
    factor = -2.0j * outgoing_longitudinal * pixel_area
    return np.asarray(values * factor, dtype=np.complex128)


def scattering_potential_to_refractive_index(
    scattering_potential: "ArrayLike",
    *,
    medium_wavenumber: float,
    medium_refractive_index: float,
) -> "NDArray[np.complex128]":
    """Convert ODTbrain's object function ``f`` to refractive index ``n``."""

    potential = np.asarray(scattering_potential, dtype=np.complex128)
    if not np.all(np.isfinite(potential)):
        raise ValueError("scattering_potential must contain only finite values")
    k_medium = float(medium_wavenumber)
    n_medium = float(medium_refractive_index)
    if not np.isfinite(k_medium) or k_medium <= 0.0:
        raise ValueError("medium_wavenumber must be finite and positive")
    if not np.isfinite(n_medium) or n_medium <= 0.0:
        raise ValueError("medium_refractive_index must be finite and positive")
    refractive_index = n_medium * np.sqrt(1.0 + potential / k_medium**2)
    negative_root = refractive_index.real < 0.0
    refractive_index[negative_root] *= -1.0
    return np.asarray(refractive_index, dtype=np.complex128)
