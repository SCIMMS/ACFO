"""Low-order Gaussian atomic-orbital and polarization helpers.

The orbital functions are homogeneous real solid harmonics multiplied by a
Gaussian radial envelope.  They are deliberately a small analytic validation
family, not an electronic-structure or resonant-scattering model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


_ORBITAL_DEGREE = {
    "s": 0,
    "p_x": 1,
    "p_y": 1,
    "p_z": 1,
    "d_xy": 2,
    "d_xz": 2,
    "d_yz": 2,
    "d_x2_y2": 2,
    "d_z2": 2,
    "f_z3": 3,
}

_ORBITAL_MODES = {
    "s": (0,),
    "p_x": (-1, 1),
    "p_y": (-1, 1),
    "p_z": (0,),
    "d_xy": (-2, 2),
    "d_xz": (-1, 1),
    "d_yz": (-1, 1),
    "d_x2_y2": (-2, 2),
    "d_z2": (0,),
    "f_z3": (0,),
}


def gaussian_orbital_names() -> tuple[str, ...]:
    """Return the supported real solid-harmonic Gaussian orbitals."""

    return tuple(_ORBITAL_DEGREE)


def gaussian_orbital_degree(name: str) -> int:
    """Return the homogeneous solid-harmonic degree of an orbital."""

    name = _validate_orbital_name(name)
    return int(_ORBITAL_DEGREE[name])


def _validate_orbital_name(name: str) -> str:
    if name not in _ORBITAL_DEGREE:
        choices = ", ".join(gaussian_orbital_names())
        raise ValueError(f"unknown orbital {name!r}; expected one of {choices}")
    return name


def gaussian_orbital_azimuthal_modes(
    name: str,
) -> "NDArray[np.int64]":
    """Return exact beta-mode support of one centered real orbital amplitude."""

    name = _validate_orbital_name(name)
    result = np.asarray(_ORBITAL_MODES[name], dtype=np.int64)
    result.setflags(write=False)
    return result


def gaussian_orbital_density_modes(name: str) -> "NDArray[np.int64]":
    """Return exact beta-mode support of the real orbital density ``psi**2``."""

    modes = gaussian_orbital_azimuthal_modes(name)
    result = np.unique((modes[:, None] + modes[None, :]).ravel())
    result.setflags(write=False)
    return result


def _solid_harmonic(
    name: str,
    x: "ArrayLike",
    y: "ArrayLike",
    z: "ArrayLike",
) -> "NDArray[np.float64]":
    name = _validate_orbital_name(name)
    x_array, y_array, z_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    if not all(np.all(np.isfinite(value)) for value in (x_array, y_array, z_array)):
        raise ValueError("orbital coordinates must be finite")
    if name == "s":
        return np.ones_like(x_array)
    if name == "p_x":
        return x_array
    if name == "p_y":
        return y_array
    if name == "p_z":
        return z_array
    if name == "d_xy":
        return x_array * y_array
    if name == "d_xz":
        return x_array * z_array
    if name == "d_yz":
        return y_array * z_array
    if name == "d_x2_y2":
        return x_array * x_array - y_array * y_array
    if name == "d_z2":
        return 2.0 * z_array * z_array - x_array * x_array - y_array * y_array
    if name == "f_z3":
        return z_array * (
            2.0 * z_array * z_array
            - 3.0 * x_array * x_array
            - 3.0 * y_array * y_array
        )
    raise AssertionError("validated orbital name was not dispatched")


def gaussian_orbital_solid_harmonic(
    name: str,
    x: "ArrayLike",
    y: "ArrayLike",
    z: "ArrayLike",
) -> "NDArray[np.float64]":
    """Evaluate the real homogeneous solid harmonic without its Gaussian."""

    return _solid_harmonic(name, x, y, z)


def gaussian_orbital_values(
    name: str,
    x: "ArrayLike",
    y: "ArrayLike",
    z: "ArrayLike",
    *,
    alpha: float,
) -> "NDArray[np.float64]":
    """Evaluate ``H_l(x,y,z) exp(-alpha r^2)`` for one real orbital."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    x_array, y_array, z_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    polynomial = _solid_harmonic(name, x_array, y_array, z_array)
    return polynomial * np.exp(-alpha * (x_array**2 + y_array**2 + z_array**2))


def gaussian_orbital_fourier(
    name: str,
    qx: "ArrayLike",
    qy: "ArrayLike",
    qz: "ArrayLike",
    *,
    alpha: float,
) -> "NDArray[np.complex128]":
    """Return the analytic positive-sign Fourier transform of one orbital.

    The convention is ``integral psi(r) exp(+i q.r) d^3r``.  For a harmonic
    homogeneous polynomial ``H_l``, Gaussian differentiation gives

    ``(pi/alpha)^(3/2) exp(-q^2/(4 alpha)) i^l H_l(q)/(2 alpha)^l``.
    """

    name = _validate_orbital_name(name)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    qx_array, qy_array, qz_array = np.broadcast_arrays(
        np.asarray(qx, dtype=np.float64),
        np.asarray(qy, dtype=np.float64),
        np.asarray(qz, dtype=np.float64),
    )
    polynomial = _solid_harmonic(name, qx_array, qy_array, qz_array)
    degree = _ORBITAL_DEGREE[name]
    envelope = (np.pi / alpha) ** 1.5 * np.exp(
        -(qx_array**2 + qy_array**2 + qz_array**2) / (4.0 * alpha)
    )
    return np.asarray(
        envelope * np.power(1j, degree) * polynomial / (2.0 * alpha) ** degree,
        dtype=np.complex128,
    )


def transverse_polarization_basis(
    wavevectors: "ArrayLike",
    *,
    reference_axis: "ArrayLike" = (0.0, 0.0, 1.0),
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Return two real orthonormal polarizations transverse to each wavevector."""

    vectors = np.asarray(wavevectors, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3 or not np.all(np.isfinite(vectors)):
        raise ValueError("wavevectors must be finite with final dimension three")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("polarization is undefined for a zero wavevector")
    directions = vectors / norms
    reference = np.asarray(reference_axis, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_axis must be a finite three-vector")
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        raise ValueError("reference_axis must be nonzero")
    reference = reference / reference_norm
    sigma = np.cross(reference, directions)
    sigma_norm = np.linalg.norm(sigma, axis=-1, keepdims=True)
    singular = sigma_norm[..., 0] < 1e-12
    if np.any(singular):
        fallback = np.cross(np.array([1.0, 0.0, 0.0]), directions[singular])
        fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
        use_y = fallback_norm[..., 0] < 1e-12
        if np.any(use_y):
            fallback[use_y] = np.cross(
                np.array([0.0, 1.0, 0.0]),
                directions[singular][use_y],
            )
        sigma[singular] = fallback
        sigma_norm = np.linalg.norm(sigma, axis=-1, keepdims=True)
    sigma = sigma / sigma_norm
    pi = np.cross(sigma, directions)
    pi = pi / np.linalg.norm(pi, axis=-1, keepdims=True)
    return sigma, pi


def polarized_scattering_amplitude(
    scalar_amplitude: "ArrayLike",
    scattering_tensor: "ArrayLike",
    incident_polarization: "ArrayLike",
    outgoing_polarization: "ArrayLike",
) -> "NDArray[np.complex128]":
    """Apply ``epsilon_out^H F epsilon_in`` to a scalar spatial amplitude.

    The polarization arrays and the leading dimensions of ``scattering_tensor``
    follow NumPy broadcasting.  A constant ``(3, 3)`` tensor is therefore
    sufficient for a whole ring stack.
    """

    amplitude = np.asarray(scalar_amplitude, dtype=np.complex128)
    tensor = np.asarray(scattering_tensor, dtype=np.complex128)
    incident = np.asarray(incident_polarization, dtype=np.complex128)
    outgoing = np.asarray(outgoing_polarization, dtype=np.complex128)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError("scattering_tensor must end in shape (3, 3)")
    if incident.ndim < 1 or incident.shape[-1] != 3:
        raise ValueError("incident_polarization must end in length three")
    if outgoing.ndim < 1 or outgoing.shape[-1] != 3:
        raise ValueError("outgoing_polarization must end in length three")
    if not all(
        np.all(np.isfinite(value))
        for value in (amplitude, tensor, incident, outgoing)
    ):
        raise ValueError("scattering inputs must be finite")
    coupling = np.einsum(
        "...a,...ab,...b->...",
        np.conj(outgoing),
        tensor,
        incident,
    )
    return np.asarray(amplitude * coupling, dtype=np.complex128)
