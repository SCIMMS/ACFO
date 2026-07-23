from __future__ import annotations

from scripts.validate_general_curvature_green_controls import run_controls
from scripts.validate_general_curvature_holdout import validate_holdouts


def test_r2_holdout_smoke_covers_precision_and_curve_stress() -> None:
    _, summary = validate_holdouts(
        seeds=(2026071401,),
        n_u=12,
        n_phi=16,
    )
    worst = summary["worst_case"]
    assert worst["complex128_forward_relative_l2"] <= 1e-11
    assert worst["complex128_weighted_adjoint_relative_l2"] <= 1e-11
    assert worst["complex128_weighted_dot_error"] <= 1e-11
    assert worst["complex64_forward_relative_l2"] <= 3e-6
    assert worst["complex64_weighted_adjoint_relative_l2"] <= 3e-6
    assert worst["complex64_weighted_dot_error"] <= 3e-6
    diagnostics = summary["curve_diagnostics"]
    assert diagnostics["near_axis"]["q_perp_min"] == 0.0
    assert diagnostics["strong_curvature"]["total_absolute_tangent_turn_rad"] > 3.0


def test_r3_green_control_smoke_detects_wrong_geometry_and_vector_controls() -> None:
    result = run_controls(n=12, n_phi=16, n_u=8)
    metrics = result["metrics"]
    assert metrics["correct_extraordinary"][
        "green_field_acfo_vs_direct_relative_l2"
    ] <= 1e-10
    assert metrics["forced_sphere_geometry"][
        "single_global_gain_relative_l2"
    ] >= 0.20
    assert metrics["axis_rotation_covariance"]["residue_relative_l2"] <= 1e-10
    assert metrics["axis_rotation_covariance"]["field_relative_l2"] <= 1e-10
    assert metrics["spatially_varying_vector_source"][
        "second_to_first_singular_value_ratio"
    ] >= 0.05
    assert metrics["spatially_varying_vector_source"][
        "green_field_acfo_vs_direct_relative_l2"
    ] <= 1e-10
    assert metrics["ordinary_selection_control"][
        "forbidden_to_allowed_field_norm_ratio"
    ] <= 1e-12
    assert metrics["simple_pole_guards"]["degenerate_optic_axis_rejected"]
    assert metrics["simple_pole_guards"]["grazing_derivative_pole_rejected"]
