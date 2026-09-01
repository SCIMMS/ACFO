"""Spectral Maxwell Green-tensor references for homogeneous anisotropic media.

The routines in this module deliberately start from the Cartesian Maxwell
wave operator rather than from the branch-specific scalar dispersion weights
used by the prepared ACFO operator.  They provide an independent first-Born
amplitude oracle at a simple outgoing pole.  Interface coupling, finite
crystals, multiple scattering, and nonlinear propagation are outside scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validated_inputs(
    k_xyz: np.ndarray,
    k0: float,
    epsilon_tensor: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    wavevectors = np.asarray(k_xyz)
    if wavevectors.ndim < 1 or wavevectors.shape[-1] != 3:
        raise ValueError("k_xyz must end in a Cartesian axis of length 3")
    if not np.all(np.isfinite(wavevectors)):
        raise ValueError("k_xyz must be finite")
    vacuum_wave_number = float(k0)
    if not np.isfinite(vacuum_wave_number) or vacuum_wave_number <= 0.0:
        raise ValueError("k0 must be finite and positive")
    epsilon = np.asarray(epsilon_tensor)
    if epsilon.shape != (3, 3) or not np.all(np.isfinite(epsilon)):
        raise ValueError("epsilon_tensor must be a finite 3 x 3 matrix")
    if not np.allclose(epsilon, np.conjugate(epsilon.T), rtol=0.0, atol=1e-13):
        raise ValueError("epsilon_tensor must be Hermitian")
    if np.min(np.linalg.eigvalsh(epsilon)) <= 0.0:
        raise ValueError("epsilon_tensor must be positive definite")
    dtype = np.result_type(wavevectors.dtype, epsilon.dtype, np.float64)
    return (
        np.asarray(wavevectors, dtype=dtype),
        vacuum_wave_number,
        np.asarray(epsilon, dtype=dtype),
    )


def maxwell_wave_operator(
    k_xyz: np.ndarray,
    *,
    k0: float,
    epsilon_tensor: np.ndarray,
) -> np.ndarray:
    """Return ``k k^T - (k . k) I + k0^2 epsilon``.

    For real wavevectors this is the frequency-domain source-free Maxwell
    operator for the ``exp(-i omega t)`` convention, up to an overall sign.
    Complex wavevectors are accepted for resolvent-limit checks; in that case
    the analytic continuation uses a transpose rather than a conjugate in the
    polynomial ``k k^T - (k . k) I``.
    """

    wavevectors, vacuum_wave_number, epsilon = _validated_inputs(
        k_xyz, k0, epsilon_tensor
    )
    outer = wavevectors[..., :, None] * wavevectors[..., None, :]
    squared = np.sum(wavevectors * wavevectors, axis=-1)
    identity = np.eye(3, dtype=outer.dtype)
    return (
        outer
        - squared[..., None, None] * identity
        + vacuum_wave_number**2 * epsilon
    )


def maxwell_wave_operator_derivative(
    k_xyz: np.ndarray,
    *,
    propagation_axis: int = 2,
) -> np.ndarray:
    """Differentiate the Cartesian Maxwell wave operator with respect to k-axis."""

    wavevectors = np.asarray(k_xyz)
    if wavevectors.ndim < 1 or wavevectors.shape[-1] != 3:
        raise ValueError("k_xyz must end in a Cartesian axis of length 3")
    if not np.all(np.isfinite(wavevectors)):
        raise ValueError("k_xyz must be finite")
    axis = int(propagation_axis)
    if axis not in (0, 1, 2):
        raise ValueError("propagation_axis must be 0, 1, or 2")
    unit = np.zeros(3, dtype=wavevectors.dtype)
    unit[axis] = 1
    identity = np.eye(3, dtype=wavevectors.dtype)
    flat = wavevectors.reshape(-1, 3)
    return (
        unit[None, :, None] * flat[:, None, :]
        + flat[:, :, None] * unit[None, None, :]
        - 2.0
        * flat[:, axis, None, None]
        * identity[None, :, :]
    ).reshape(wavevectors.shape[:-1] + (3, 3))


def maxwell_spectral_residue(
    k_xyz: np.ndarray,
    *,
    k0: float,
    epsilon_tensor: np.ndarray,
    propagation_axis: int = 2,
    singular_tolerance: float = 1e-10,
    degeneracy_tolerance: float = 1e-10,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Return the inverse-Maxwell-operator residue at simple real poles.

    If ``v`` is the normalized null eigenvector of ``M(k)``, the residue of
    ``M(k)^{-1}`` with respect to the selected wave-vector coordinate is
    ``v v^H / (v^H (dM/dk_axis) v)``.  This normalization is obtained directly
    from the dyadic Maxwell operator and must not be replaced by a scalar-PDE
    pole weight after normalizing the electric eigenpolarization.
    """

    wavevectors, vacuum_wave_number, epsilon = _validated_inputs(
        k_xyz, k0, epsilon_tensor
    )
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance <= 0.0:
        raise ValueError("degeneracy_tolerance must be finite and positive")
    if np.iscomplexobj(wavevectors) and np.max(np.abs(np.imag(wavevectors))) > 0.0:
        raise ValueError("spectral residues require real pole wavevectors")
    wavevectors = np.asarray(np.real(wavevectors), dtype=np.float64)
    operator = maxwell_wave_operator(
        wavevectors, k0=vacuum_wave_number, epsilon_tensor=epsilon
    )
    derivative = maxwell_wave_operator_derivative(
        wavevectors, propagation_axis=propagation_axis
    )
    flat_operator = operator.reshape(-1, 3, 3)
    flat_derivative = derivative.reshape(-1, 3, 3)
    residues = np.empty_like(flat_operator, dtype=np.complex128)
    residuals = np.empty(flat_operator.shape[0], dtype=np.float64)
    denominators = np.empty(flat_operator.shape[0], dtype=np.float64)
    spectral_gaps = np.empty(flat_operator.shape[0], dtype=np.float64)
    for index, (matrix, matrix_derivative) in enumerate(
        zip(flat_operator, flat_derivative, strict=True)
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        mode = int(np.argmin(np.abs(eigenvalues)))
        vector = np.asarray(eigenvectors[:, mode], dtype=np.complex128)
        scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
        residual = float(abs(eigenvalues[mode]) / scale)
        if residual > float(singular_tolerance):
            raise ValueError(
                f"wavevector is not on a simple Maxwell pole: residual={residual:.3e}"
            )
        ordered_absolute_eigenvalues = np.sort(np.abs(eigenvalues))
        spectral_gap = float(ordered_absolute_eigenvalues[1] / scale)
        if spectral_gap <= float(degeneracy_tolerance):
            raise ValueError(
                "wavevector is not on a simple Maxwell pole: "
                f"normalized spectral gap={spectral_gap:.3e}"
            )
        denominator = float(
            np.real(np.vdot(vector, np.asarray(matrix_derivative) @ vector))
        )
        if abs(denominator) <= 1e-14 * scale:
            raise ValueError("Maxwell pole is not simple along propagation_axis")
        residues[index] = np.outer(vector, np.conjugate(vector)) / denominator
        residuals[index] = residual
        denominators[index] = denominator
        spectral_gaps[index] = spectral_gap
    shaped = residues.reshape(wavevectors.shape[:-1] + (3, 3))
    if not return_diagnostics:
        return shaped
    diagnostics = {
        "max_normalized_null_residual": float(np.max(residuals)),
        "min_normalized_spectral_gap": float(np.min(spectral_gaps)),
        "min_abs_pole_derivative": float(np.min(np.abs(denominators))),
        "max_abs_pole_derivative": float(np.max(np.abs(denominators))),
    }
    return shaped, diagnostics


def maxwell_resolvent_residue(
    k_xyz: np.ndarray,
    *,
    k0: float,
    epsilon_tensor: np.ndarray,
    eta: float,
    propagation_axis: int = 2,
) -> np.ndarray:
    """Approximate a pole residue from ``delta * M(k + delta e_axis)^-1``."""

    wavevectors, vacuum_wave_number, epsilon = _validated_inputs(
        k_xyz, k0, epsilon_tensor
    )
    axis = int(propagation_axis)
    if axis not in (0, 1, 2):
        raise ValueError("propagation_axis must be 0, 1, or 2")
    width = float(eta)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("eta must be finite and positive")
    shifted = np.asarray(wavevectors, dtype=np.complex128).copy()
    delta = 1j * width
    shifted[..., axis] += delta
    operator = maxwell_wave_operator(
        shifted, k0=vacuum_wave_number, epsilon_tensor=epsilon
    )
    return delta * np.linalg.inv(operator)


def apply_maxwell_spectral_residue(
    scalar_amplitude: np.ndarray,
    k_xyz: np.ndarray,
    source_vector: np.ndarray,
    *,
    k0: float,
    epsilon_tensor: np.ndarray,
    propagation_axis: int = 2,
) -> np.ndarray:
    """Apply the dyadic pole residue to a Fourier amplitude and source vector."""

    amplitude = np.asarray(scalar_amplitude, dtype=np.complex128)
    wavevectors = np.asarray(k_xyz)
    source = np.asarray(source_vector, dtype=np.complex128)
    if wavevectors.shape != amplitude.shape + (3,):
        raise ValueError("k_xyz shape must equal scalar_amplitude.shape + (3,)")
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_vector must be a finite complex vector of shape (3,)")
    residue = maxwell_spectral_residue(
        wavevectors,
        k0=k0,
        epsilon_tensor=epsilon_tensor,
        propagation_axis=propagation_axis,
    )
    projected = np.einsum("...ij,j->...i", residue, source)
    return amplitude[..., None] * projected
