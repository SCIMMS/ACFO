from __future__ import annotations

import numpy as np
import pytest

from waxs_cake.solvers import normalize_form_factors
from waxs_cake.xray_form_factors import xray_f0, xray_f0_form_factors


def test_xray_f0_uses_solver_inverse_nm_units() -> None:
    q_inv_nm = np.array([0.0, 10.0, 20.0])
    values = xray_f0("C", q_inv_nm)

    assert values.shape == q_inv_nm.shape
    assert values[0] == pytest.approx(6.0, rel=1e-3)
    assert values[0] > values[1] > values[2]


def test_xray_f0_form_factors_are_solver_compatible() -> None:
    q_inv_nm = np.array([0.0, 5.0, 10.0])
    form_factors = xray_f0_form_factors(np.array(["O", "C", "O"]), q_inv_nm)
    normalized = normalize_form_factors(["C", "O"], q_inv_nm, form_factors)

    assert normalized.shape == (2, q_inv_nm.size)
    np.testing.assert_allclose(normalized[0], xray_f0("C", q_inv_nm))
    np.testing.assert_allclose(normalized[1], xray_f0("O", q_inv_nm))
