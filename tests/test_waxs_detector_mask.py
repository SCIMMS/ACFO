from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_protein_nanocrystal_finufft_fair import (  # noqa: E402
    detector_rectangle_mask,
    masked_ring_mean,
    masked_row_relative_l2,
)


def test_rectangular_detector_keeps_low_q_and_partial_outer_arc() -> None:
    q = np.array([0.05, 6.3])
    phi = np.linspace(0.0, 2.0 * np.pi, 2160, endpoint=False)
    mask = detector_rectangle_mask(
        q,
        phi,
        wavelength_nm=0.08,
        active_width_mm=155.1,
        active_height_mm=162.15,
        distance_mm=100.0,
    )

    assert np.all(mask[0])
    assert 0 < np.count_nonzero(mask[1]) < phi.size


def test_masked_metrics_ignore_inactive_nodes_and_empty_rows() -> None:
    mask = np.array([[True, False], [False, False], [True, True]])
    reference = np.array([[2.0, 100.0], [20.0, 30.0], [1.0, 3.0]])
    values = np.array([[2.0, -50.0], [-20.0, -30.0], [1.0, 3.0]])

    np.testing.assert_allclose(masked_ring_mean(values, mask), np.array([2.0, 0.0, 2.0]))
    np.testing.assert_allclose(masked_row_relative_l2(values, reference, mask), 0.0)
