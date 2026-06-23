from __future__ import annotations

from waxs_cake import choose_physical_grid, water_box_side_nm


def test_water_box_side_for_one_million_atoms_is_about_21_nm() -> None:
    side_nm = water_box_side_nm(1_000_000)

    assert 21.0 < side_nm < 22.0


def test_physical_grid_scales_bins_from_resolution() -> None:
    grid = choose_physical_grid(
        1_000_000,
        bin_width_nm=0.1,
        qmax=2.2,
        q_unit="inv_angstrom",
        n_phi_detector=180,
        harmonic_margin=16,
    )

    assert 21.0 < grid.box_side_nm < 22.0
    assert grid.n_z >= 215
    assert grid.n_r >= 152
    assert grid.n_phi >= grid.n_phi_detector
    assert grid.n_phi >= grid.n_phi_bandlimit
    assert grid.dr_nm <= 0.1
    assert grid.dz_nm <= 0.1


def test_physical_grid_rounds_phi_to_fft_friendly_length() -> None:
    grid = choose_physical_grid(
        500_000,
        bin_width_nm=0.1,
        qmax=6.3,
        q_unit="inv_angstrom",
        n_phi_detector=180,
        harmonic_margin=16,
    )

    assert grid.n_phi_bandlimit == 1556
    assert grid.n_phi == 1600


def test_physical_grid_arc_rule_limits_outer_arc_length() -> None:
    grid = choose_physical_grid(
        100_000,
        bin_width_nm=0.2,
        qmax=2.2,
        q_unit="inv_angstrom",
        n_phi_detector=180,
        angular_rule="arc",
    )

    assert grid.n_phi >= grid.n_phi_arc
    assert grid.outer_arc_nm <= 0.2
