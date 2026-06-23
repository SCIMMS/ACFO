from __future__ import annotations

import numpy as np

from scripts.validate_public_waxs_structures import debye_weighted_pdist


def test_debye_weighted_pdist_accepts_q_dependent_form_factors() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    elements = np.array(["C", "O"])
    q = np.array([1.0, 3.0], dtype=np.float64)
    form_factors = {
        "C": np.array([6.0, 5.0]),
        "O": np.array([8.0, 7.0]),
    }

    got = debye_weighted_pdist(coords, elements, q, form_factors)
    expected = np.array(
        [
            6.0**2 + 8.0**2 + 2.0 * 6.0 * 8.0 * np.sinc(q[0] * 0.2 / np.pi),
            5.0**2 + 7.0**2 + 2.0 * 5.0 * 7.0 * np.sinc(q[1] * 0.2 / np.pi),
        ]
    )

    np.testing.assert_allclose(got, expected)
