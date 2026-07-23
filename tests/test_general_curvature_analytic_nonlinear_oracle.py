from __future__ import annotations

from scripts.validate_general_curvature_analytic_nonlinear_oracle import (
    chi2_factorization_relative_l2,
    evaluate_level,
    make_physical_problem,
)


def test_gaussian_pump_chi2_factorization_is_machine_precision() -> None:
    problem = make_physical_problem()
    assert chi2_factorization_relative_l2(problem) <= 1e-13


def test_reduced_analytic_nonlinear_green_operator_matches_direct_sum() -> None:
    problem = make_physical_problem()
    level, _ = evaluate_level(
        problem,
        n_xyz=12,
        n_r=18,
        n_phi=36,
        n_u=5,
        u_min=0.10,
        u_max=0.72,
        direct_operator=True,
    )
    for branch in ("ordinary", "extraordinary"):
        row = level["branches"][branch]
        assert row["scalar_acfo_vs_binned_direct"]["relative_l2"] <= 1e-10
        assert row["green_acfo_vs_binned_direct"]["relative_l2"] <= 1e-10
        assert row["maxwell_pole"]["max_normalized_null_residual"] <= 1e-12
