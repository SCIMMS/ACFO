from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_giwaxs_dwba_multilayer import (  # noqa: E402
    _field_product_for_targets,
    binned_dwba_direct_amplitude,
    build_distorted_wave_stack,
    build_prepared_dwba_geometry,
    direct_dwba_atom_amplitude,
    dwba_field_product_grid,
    execute_prepared_dwba_geometry,
    finufft_dwba_channel_amplitude,
    make_giwaxs_detector,
    make_synthetic_multilayer,
)
from waxs_cake import make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import relative_l2  # noqa: E402


def _small_detector():
    return make_giwaxs_detector(
        wavelength_nm=0.15406,
        alpha_i_deg=0.2,
        alpha_f_min_deg=0.3,
        alpha_f_max_deg=3.0,
        n_alpha_f=4,
        two_theta_min_deg=-3.0,
        two_theta_max_deg=3.0,
        n_two_theta=5,
    )


def test_dwba_field_grid_uses_the_same_recursion_as_direct_fields():
    detector = _small_detector()
    _, _, z_edges = make_synthetic_multilayer(
        n_atoms=90,
        n_layers=3,
        radius_nm=1.0,
        layer_thickness_nm=1.2,
        seed=10,
    )
    stack = build_distorted_wave_stack(detector, z_edges)
    z_probe = np.linspace(z_edges[0] + 0.1, z_edges[-1] - 0.1, 7)
    grid = dwba_field_product_grid(stack, z_probe)

    layer_ids = np.array([0, 1, 2, 0, 1, 2, 1], dtype=np.int64)
    direct = _field_product_for_targets(
        stack,
        z_probe,
        layer_ids,
        0,
        detector.qx.size,
    )
    for item, layer in enumerate(layer_ids):
        np.testing.assert_allclose(grid[:, layer, item], direct[:, item], rtol=1e-12, atol=1e-12)


def test_prepared_dwba_contraction_matches_binned_direct_sum():
    detector = _small_detector()
    coords, layer_ids, z_edges = make_synthetic_multilayer(
        n_atoms=700,
        n_layers=3,
        radius_nm=1.4,
        layer_thickness_nm=1.0,
        seed=33,
    )
    binned = make_cylindrical_histogram(
        coords,
        element_indices=layer_ids,
        n_elements=3,
        n_r=10,
        n_z=16,
        n_phi=160,
        r_max=1.4,
        z_range=(float(z_edges[0]), float(z_edges[-1])),
        backend="numpy",
        hist_dtype=np.float64,
    )
    stack = build_distorted_wave_stack(detector, z_edges)
    field_grid = dwba_field_product_grid(stack, binned.z_centers)
    direct_bins = binned_dwba_direct_amplitude(
        binned,
        field_grid,
        detector.qx,
        detector.qy,
        target_chunk=6,
    )
    geometry = build_prepared_dwba_geometry(
        binned,
        detector.qx,
        detector.qy,
        max_mode=79,
    )
    prepared = execute_prepared_dwba_geometry(
        binned,
        geometry,
        field_grid,
        detector.shape,
        target_chunk=6,
    )

    assert relative_l2(prepared, direct_bins) < 1e-7

    pruned_geometry = build_prepared_dwba_geometry(
        binned,
        detector.qx,
        detector.qy,
        max_mode=79,
        enable_mode_pruning=True,
        mode_pruning_margin=32,
        mode_pruning_bin_size=1,
    )
    pruned = execute_prepared_dwba_geometry(
        binned,
        pruned_geometry,
        field_grid,
        detector.shape,
        target_chunk=6,
    )

    assert pruned_geometry.mode_pruning
    assert pruned_geometry.max_mode < pruned_geometry.requested_max_mode
    assert pruned_geometry.mode_work_fraction < 1.0
    assert relative_l2(pruned, direct_bins) < 1e-7


def test_atom_direct_dwba_keeps_binning_error_separate_from_contraction_error():
    detector = _small_detector()
    coords, layer_ids, z_edges = make_synthetic_multilayer(
        n_atoms=500,
        n_layers=2,
        radius_nm=1.2,
        layer_thickness_nm=0.9,
        seed=44,
    )
    binned = make_cylindrical_histogram(
        coords,
        element_indices=layer_ids,
        n_elements=2,
        n_r=12,
        n_z=18,
        n_phi=128,
        r_max=1.2,
        z_range=(float(z_edges[0]), float(z_edges[-1])),
        backend="numpy",
        hist_dtype=np.float64,
    )
    stack = build_distorted_wave_stack(detector, z_edges)
    field_grid = dwba_field_product_grid(stack, binned.z_centers)
    direct_bins = binned_dwba_direct_amplitude(
        binned,
        field_grid,
        detector.qx,
        detector.qy,
        target_chunk=6,
    )
    direct_atoms = direct_dwba_atom_amplitude(
        coords,
        layer_ids,
        stack,
        detector.qx,
        detector.qy,
        target_chunk=6,
    )

    assert relative_l2(direct_bins, direct_atoms) > 0.0
    assert relative_l2(direct_bins, direct_atoms) < 0.2


def test_finufft_dwba_channel_sum_matches_atom_direct_when_qz_is_real():
    pytest.importorskip("finufft")
    detector = _small_detector()
    coords, layer_ids, z_edges = make_synthetic_multilayer(
        n_atoms=160,
        n_layers=2,
        radius_nm=1.0,
        layer_thickness_nm=0.8,
        seed=55,
    )
    stack = build_distorted_wave_stack(detector, z_edges, absorption_imag=0.0)
    direct_atoms = direct_dwba_atom_amplitude(
        coords,
        layer_ids,
        stack,
        detector.qx,
        detector.qy,
        target_chunk=5,
    )
    finufft_atoms = finufft_dwba_channel_amplitude(
        coords,
        layer_ids,
        stack,
        detector.qx,
        detector.qy,
        eps=1e-9,
    )

    assert relative_l2(finufft_atoms, direct_atoms) < 1e-8
