from __future__ import annotations

import numpy as np

from waxs_cake import PreparedAxisymmetricOperator, estimate_bessel_cutoff
from scripts.validate_axisymmetric_harmonic_cutoff import (
    active_radius_max,
    angular_interpolation_error,
    matched_curvature_family,
    row_truncation_error,
)
from scripts.validate_axisymmetric_manifold_discrete import make_validation_object


def test_full_cutoff_matches_untruncated_forward() -> None:
    histogram = make_validation_object(n_phi=128)
    r_active = active_radius_max(histogram)
    manifold = matched_curvature_family(8.0, r_active, n_u=8)["spline"]
    operator = PreparedAxisymmetricOperator(histogram, manifold)
    full = operator.forward(histogram.hist)
    truncated = operator.forward_harmonic_cutoff(histogram.hist, histogram.n_phi // 2)
    assert np.allclose(truncated, full, rtol=0.0, atol=0.0)


def test_truncation_error_is_nonincreasing() -> None:
    histogram = make_validation_object(n_phi=128)
    r_active = active_radius_max(histogram)
    manifold = matched_curvature_family(12.0, r_active, n_u=8)["paraboloid"]
    operator = PreparedAxisymmetricOperator(histogram, manifold)
    data_fft = operator.forward_fourier(histogram.hist)
    row = int(np.argmax(manifold.q_perp))
    errors = [
        row_truncation_error(data_fft, row, operator.angular_modes, max_h)
        for max_h in range(histogram.n_phi // 2 + 1)
    ]
    assert np.all(np.diff(errors) <= 1e-15)


def test_matched_family_preserves_qperp_radius_product() -> None:
    histogram = make_validation_object(n_phi=128)
    r_active = active_radius_max(histogram)
    for manifold in matched_curvature_family(10.0, r_active).values():
        assert np.isclose(np.max(manifold.q_perp) * r_active, 10.0, rtol=0.0, atol=1e-12)


def test_bessel_cutoff_gives_alias_safe_even_grid() -> None:
    x_product = 12.0
    tolerance = 1e-8
    max_h = estimate_bessel_cutoff(x_product, tol=tolerance)
    safe_n_phi = 2 * max_h + 2
    assert angular_interpolation_error(x_product, safe_n_phi) < tolerance
    assert angular_interpolation_error(x_product, 2 * int(x_product)) > 1e-2
