from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    repeated_block_translations,
    translation_lattice_factor,
    translation_lattice_factor_separable,
)


def test_repeated_block_translations_and_lattice_factor() -> None:
    unit = np.asarray([[0.1, -0.2, 0.3], [0.7, 0.4, -0.1]])
    translations = np.asarray([[0.0, 0.0, 0.0], [1.2, -0.4, 0.8], [-0.3, 0.5, 1.1]])
    supercell = np.concatenate([unit + value for value in translations], axis=0)

    got, residual = repeated_block_translations(unit, supercell)
    assert residual < 1e-15
    assert np.allclose(got, translations)

    q = np.asarray([[0.2, 0.3, -0.1], [1.1, -0.4, 0.7]])
    expected = np.sum(np.exp(1j * (translations @ q.T)), axis=0)
    factor = translation_lattice_factor(q[:, 0], q[:, 1], q[:, 2], translations)
    assert np.allclose(factor, expected)


def test_separable_lattice_factor_matches_direct_for_skew_grid() -> None:
    basis = np.array(
        [[1.1, 0.2, -0.1], [0.3, 1.4, 0.25], [-0.2, 0.15, 0.9]]
    )
    shape = (3, 2, 4)
    origin = np.array([-1.7, 0.4, -0.8])
    translations = np.asarray(
        [
            origin + i * basis[0] + j * basis[1] + k * basis[2]
            for i in range(shape[0])
            for j in range(shape[1])
            for k in range(shape[2])
        ]
    )
    rng = np.random.default_rng(20260718)
    q = rng.normal(size=(37, 3))
    direct = translation_lattice_factor(
        q[:, 0], q[:, 1], q[:, 2], translations
    )
    separable = translation_lattice_factor_separable(
        q[:, 0], q[:, 1], q[:, 2], translations, shape
    )
    assert np.linalg.norm(separable - direct) / np.linalg.norm(direct) < 1e-14


def test_repeated_block_translations_rejects_nonrepeated_structure() -> None:
    unit = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1]])
    bad = np.concatenate([unit, unit + np.asarray([1.0, 0.0, 0.0])], axis=0)
    bad[-1, 2] += 1e-3
    with pytest.raises(ValueError, match="not an ordered exact repetition"):
        repeated_block_translations(unit, bad, atol=1e-6)
