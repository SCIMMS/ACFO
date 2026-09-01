"""Exact translation-lattice helpers for repeated finite crystals."""

from __future__ import annotations

import numpy as np


def repeated_block_translations(
    unit_coords: np.ndarray,
    supercell_coords: np.ndarray,
    *,
    atol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Extract translations from an ordered repetition of one coordinate block.

    Returns the mean translation of each block and the maximum absolute
    coordinate residual after removing those translations. A non-repeated or
    reordered structure raises ``ValueError`` rather than silently applying a
    crystallographic factorization.
    """

    unit = np.asarray(unit_coords, dtype=np.float64)
    supercell = np.asarray(supercell_coords, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != 3:
        raise ValueError("unit_coords must have shape (n_unit, 3)")
    if supercell.ndim != 2 or supercell.shape[1] != 3:
        raise ValueError("supercell_coords must have shape (n_total, 3)")
    if unit.shape[0] == 0 or supercell.shape[0] % unit.shape[0]:
        raise ValueError("supercell size must be an integer multiple of unit size")
    blocks = supercell.reshape(-1, unit.shape[0], 3)
    offsets = blocks - unit[None, :, :]
    translations = np.mean(offsets, axis=1)
    residual = float(
        np.max(np.abs(offsets - translations[:, None, :]), initial=0.0)
    )
    if residual > atol:
        raise ValueError(
            f"supercell is not an ordered exact repetition: residual {residual:.3e} > {atol:.3e}"
        )
    return np.ascontiguousarray(translations), residual


def translation_lattice_factor(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    translations: np.ndarray,
    *,
    target_chunk_size: int = 4096,
) -> np.ndarray:
    """Return ``sum_t exp(i q.target dot t)`` for arbitrary target vectors."""

    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    qz = np.asarray(qz, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if qx.ndim != 1 or qy.shape != qx.shape or qz.shape != qx.shape:
        raise ValueError("qx, qy, and qz must be one-dimensional with equal shape")
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must have shape (n_translation, 3)")
    target_chunk_size = int(target_chunk_size)
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive")
    out = np.empty(qx.size, dtype=np.complex128)
    for start in range(0, qx.size, target_chunk_size):
        stop = min(start + target_chunk_size, qx.size)
        phase = (
            translations[:, 0, None] * qx[None, start:stop]
            + translations[:, 1, None] * qy[None, start:stop]
            + translations[:, 2, None] * qz[None, start:stop]
        )
        out[start:stop] = np.sum(np.exp(1j * phase), axis=0)
    return out


def translation_lattice_factor_separable(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    translations: np.ndarray,
    supercell_shape: tuple[int, int, int] | list[int],
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Return an exact finite-lattice factor for a separable 3-D grid.

    The translation vectors may form a skew or rotated parallelepiped.  The
    ordered translation list is validated against ``supercell_shape`` before
    the three finite geometric sums are multiplied.  Nonseparable inputs raise
    rather than silently changing the result.
    """

    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    qz = np.asarray(qz, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    shape = tuple(int(value) for value in supercell_shape)
    if qx.ndim != 1 or qy.shape != qx.shape or qz.shape != qx.shape:
        raise ValueError("qx, qy, and qz must be one-dimensional with equal shape")
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must have shape (n_translation, 3)")
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("supercell_shape must contain three positive integers")
    if int(np.prod(shape)) != translations.shape[0]:
        raise ValueError("supercell_shape product does not match translations")

    grid = translations.reshape(*shape, 3)
    origin = grid[0, 0, 0]
    basis = np.zeros((3, 3), dtype=np.float64)
    if shape[0] > 1:
        basis[0] = grid[1, 0, 0] - origin
    if shape[1] > 1:
        basis[1] = grid[0, 1, 0] - origin
    if shape[2] > 1:
        basis[2] = grid[0, 0, 1] - origin
    indices = np.indices(shape, dtype=np.float64)
    expected = (
        origin[None, None, None, :]
        + indices[0][..., None] * basis[0]
        + indices[1][..., None] * basis[1]
        + indices[2][..., None] * basis[2]
    )
    residual = float(np.max(np.abs(grid - expected), initial=0.0))
    if residual > atol:
        raise ValueError(
            f"translations are not an ordered separable lattice: residual {residual:.3e} > {atol:.3e}"
        )

    q = np.column_stack((qx, qy, qz))
    out = np.exp(1j * (q @ origin))
    for count, vector in zip(shape, basis):
        if count == 1:
            continue
        step = np.exp(1j * (q @ vector))
        term = np.ones(qx.size, dtype=np.complex128)
        finite_sum = term.copy()
        for _ in range(1, count):
            term = term * step
            finite_sum += term
        out *= finite_sum
    return out
