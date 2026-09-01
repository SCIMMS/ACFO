from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waxs_cake import (
    PreparedCakePlan,
    circular_fft_amplitude,
    cylindrical_flat_indices,
    direct_amplitude,
    jacobi_anger_amplitude,
    make_cylindrical_histogram,
    make_sparse_cylindrical_histogram_from_flat_indices,
    nufft_amplitude,
    nufft_amplitude_chunked,
)
from waxs_cake.metrics import relative_l2


def test_jacobi_matches_direct_for_center_atom() -> None:
    coords = np.array([[0.0, 0.0, 0.0]])
    q = np.linspace(0.2, 2.0, 4)
    phi = (np.arange(32) + 0.5) * (2 * np.pi / 32)
    binned = make_cylindrical_histogram(
        coords,
        n_r=1,
        n_z=1,
        n_phi=32,
        r_max=1e-9,
        z_range=(-0.5, 0.5),
    )

    ref = direct_amplitude(coords, q, 1.0, phi)
    got, _ = jacobi_anger_amplitude(binned, q, 1.0, harmonic_cutoff=0)

    assert relative_l2(got, ref) < 1e-12


def test_circular_and_jacobi_agree_on_binned_structure() -> None:
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(100, 3))
    q = np.linspace(0.2, 1.0, 5)
    binned = make_cylindrical_histogram(coords, n_r=10, n_z=10, n_phi=64)

    circ = circular_fft_amplitude(binned, q, 1.0)
    jac, _ = jacobi_anger_amplitude(binned, q, 1.0, harmonic_cutoff=32)

    assert relative_l2(jac, circ) < 1e-8


def test_prepared_plan_cached_kernel_matches_uncached() -> None:
    rng = np.random.default_rng(9)
    coords = rng.normal(size=(200, 3))
    q = np.linspace(0.1, 1.5, 7)
    binned = make_cylindrical_histogram(coords, n_r=12, n_z=12, n_phi=72)

    uncached = PreparedCakePlan(binned, q, 1.0).circular_fft()
    cached = PreparedCakePlan(binned, q, 1.0, cache_kernel_fft=True).circular_fft()
    hybrid, _, _ = PreparedCakePlan(binned, q, 1.0).hybrid()

    assert relative_l2(cached, uncached) < 1e-12
    assert relative_l2(hybrid, uncached) < 1e-8


def test_kernel_interpolation_approximates_exact_kernel_path() -> None:
    rng = np.random.default_rng(10)
    coords = rng.normal(size=(300, 3))
    q = np.linspace(0.1, 0.8, 6)
    binned = make_cylindrical_histogram(coords, n_r=10, n_z=10, n_phi=64)

    exact = PreparedCakePlan(binned, q, 1.0).circular_fft()
    interpolated = PreparedCakePlan(
        binned,
        q,
        1.0,
        kernel_interpolation_dx=0.005,
    ).circular_fft()

    assert relative_l2(interpolated, exact) < 1e-6


def test_z_reduced_cache_and_form_factor_replacement() -> None:
    rng = np.random.default_rng(11)
    coords = rng.normal(size=(500, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(coords, n_r=12, n_z=12, n_phi=64)
    ff = {"X": 1.0 + 0.2 * q}

    plan = PreparedCakePlan(binned, q, 1.0)
    expected = PreparedCakePlan(binned, q, 1.0, form_factors=ff).circular_fft()
    got = plan.circular_fft_with_form_factors(ff)

    plan.precompute_z_reduced()
    cached = plan.circular_fft_with_form_factors(ff)

    assert relative_l2(got, expected) < 1e-12
    assert relative_l2(cached, expected) < 1e-12


def test_ring_average_intensity_matches_spatial_average() -> None:
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(400, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(coords, n_r=12, n_z=12, n_phi=64)
    plan = PreparedCakePlan(binned, q, 1.0)

    amp = plan.circular_fft()
    expected = np.mean(np.abs(amp) ** 2, axis=1)
    got = plan.ring_average_intensity()

    assert relative_l2(got, expected) < 1e-12


def test_r_grouped_ring_average_matches_spatial_average() -> None:
    rng = np.random.default_rng(121)
    coords = rng.normal(size=(450, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=72,
        hist_dtype=np.float32,
    )
    plan = PreparedCakePlan(binned, q, 1.0, complex_dtype=np.complex64)

    amp = plan.circular_fft()
    expected = np.mean(np.abs(amp) ** 2, axis=1)
    got = plan.ring_average_intensity_r_grouped()

    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_bandlimit_large_margin_matches_r_grouped_curve() -> None:
    rng = np.random.default_rng(122)
    coords = rng.normal(size=(450, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=72,
        hist_dtype=np.float32,
    )
    plan = PreparedCakePlan(binned, q, 1.0, complex_dtype=np.complex64)

    expected = plan.ring_average_intensity_r_grouped()
    got = plan.ring_average_intensity_r_dependent_bandlimit(
        margin=10_000,
        cutoff_bin_size=8,
    )

    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_cake_large_margin_matches_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(123)
    coords = rng.normal(size=(500, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=72,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = plan.circular_fft()
    got = plan.circular_fft_r_dependent_bandlimit(
        margin=10_000,
        cutoff_bin_size=8,
    )

    assert relative_l2(got, expected) < 5e-6


def test_analytic_kernel_hat_modes_matches_sampled_fft() -> None:
    coords = np.array([[0.1, 0.2, 0.0]])
    q = np.linspace(0.05, 0.4, 5)
    binned = make_cylindrical_histogram(
        coords,
        n_r=6,
        n_z=3,
        n_phi=128,
        r_max=1.0,
        z_range=(-0.5, 0.5),
    )
    plan = PreparedCakePlan(binned, q, 1.0)
    max_cutoff = 20

    sampled = plan._kernel_hat_block(np.arange(q.size))[:, :, : max_cutoff + 1]
    analytic = plan._analytic_kernel_hat_modes(np.arange(q.size), max_cutoff)

    assert relative_l2(analytic, sampled) < 1e-10


def test_table_analytic_kernel_hat_modes_matches_miller() -> None:
    coords = np.array([[0.1, 0.2, 0.0]])
    q = np.linspace(0.05, 0.4, 5)
    binned = make_cylindrical_histogram(
        coords,
        n_r=6,
        n_z=3,
        n_phi=128,
        r_max=1.0,
        z_range=(-0.5, 0.5),
    )
    plan = PreparedCakePlan(binned, q, 1.0)
    max_cutoff = 20

    expected = plan._analytic_kernel_hat_modes(np.arange(q.size), max_cutoff)
    got = plan._analytic_kernel_hat_modes(
        np.arange(q.size),
        max_cutoff,
        table_dx=0.005,
    )

    assert relative_l2(got, expected) < 1e-8


def test_r_dependent_cake_analytic_kernel_matches_sampled_kernel() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(124)
    coords = rng.normal(scale=0.4, size=(500, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=False,
    )
    got = plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )

    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_cake_half_spectrum_matches_full_spectrum() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(125)
    coords = rng.normal(scale=0.45, size=(600, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    complex_binned = replace(binned, hist=binned.hist.astype(np.complex64))

    real_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )
    complex_plan = PreparedCakePlan(
        complex_binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    got = real_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )
    expected = complex_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )

    assert real_plan._hhat is None
    assert real_plan._hhat_half is not None
    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_cake_reuses_full_fft_positive_modes_without_compact_copy() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(1251)
    coords = rng.normal(scale=0.45, size=(500, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    expected_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )
    got_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = expected_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )
    _ = got_plan.circular_fft()
    got = got_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )

    assert got_plan._hhat is not None
    assert got_plan._hhat_half is None
    assert not got_plan._hhat_half_mode_cache
    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_cake_r_block_streaming_matches_analytic_half_spectrum() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(1252)
    coords = rng.normal(scale=0.45, size=(600, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=14,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    expected_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )
    got_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = expected_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )
    got = got_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
        r_block_size=5,
    )

    assert got_plan._hhat is None
    assert got_plan._hhat_half is None
    assert relative_l2(got, expected) < 5e-6


def test_r_dependent_cake_fused_miller_matches_analytic_half_spectrum() -> None:
    cpp_solvers = pytest.importorskip("waxs_cake._cpp_solvers")
    assert hasattr(cpp_solvers, "circular_contract_r_dependent_half_modes_miller64")

    rng = np.random.default_rng(1253)
    coords = rng.normal(scale=0.45, size=(600, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=14,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    expected_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )
    got_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )
    got_stream_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = expected_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )
    got = got_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
        fused_analytic_kernel=True,
    )
    got_stream = got_stream_plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
        fused_analytic_kernel=True,
        r_block_size=5,
    )

    assert relative_l2(got, expected) < 5e-6
    assert relative_l2(got_stream, expected) < 5e-6


def test_r_dependent_cake_z_projection_matches_half_spectrum() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(126)
    coords = rng.normal(scale=0.45, size=(600, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=13,
        n_z=11,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    )

    expected = plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
    )
    got = plan.circular_fft_r_dependent_bandlimit(
        margin=16,
        cutoff_bin_size=8,
        analytic_kernel=True,
        z_projection=True,
    )

    assert relative_l2(got, expected) < 5e-6


def test_chunked_nufft_matches_single_call() -> None:
    pytest.importorskip("finufft")

    rng = np.random.default_rng(127)
    coords = rng.normal(scale=0.3, size=(40, 3))
    q = np.linspace(0.05, 0.5, 7)
    phi = (np.arange(18) + 0.5) * (2.0 * np.pi / 18)

    expected = nufft_amplitude(coords, q, 1.0, phi, eps=1e-10)
    got = nufft_amplitude_chunked(coords, q, 1.0, phi, eps=1e-10, q_block_size=2)

    assert relative_l2(got, expected) < 1e-12


def test_cpp_circular_backend_matches_numpy_backend() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(13)
    coords = rng.normal(size=(800, 3))
    q = np.linspace(0.1, 1.4, 9)
    binned = make_cylindrical_histogram(coords, n_r=14, n_z=13, n_phi=72)

    expected = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="numpy",
    ).circular_fft()
    got = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
    ).circular_fft()

    assert relative_l2(got, expected) < 1e-12


def test_cpp_circular_float32_complex64_block_z_matches_numpy() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(131)
    coords = rng.normal(size=(800, 3))
    q = np.linspace(0.1, 1.4, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=14,
        n_z=13,
        n_phi=72,
        hist_dtype=np.float32,
        backend="cpp",
    )

    expected = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="numpy",
        complex_dtype=np.complex64,
    ).circular_fft()
    got = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
    ).circular_fft()

    assert relative_l2(got, expected) < 5e-6


def test_cpp_circular_backend_matches_numpy_with_z_cache_and_form_factors() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(14)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.5, 10)
    binned = make_cylindrical_histogram(coords, n_r=15, n_z=14, n_phi=80)
    ff = {"X": (1.0 + 0.1 * q) * np.exp(0.03j * q)}

    expected_plan = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy")
    expected_plan.precompute_z_reduced()
    expected = expected_plan.circular_fft_with_form_factors(ff)

    got_plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")
    got_plan.precompute_z_reduced()
    got = got_plan.circular_fft_with_form_factors(ff)

    assert relative_l2(got, expected) < 1e-12


def test_harmonic_bandlimit_large_margin_matches_exact_circular() -> None:
    rng = np.random.default_rng(15)
    coords = rng.normal(size=(500, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(coords, n_r=12, n_z=12, n_phi=64)

    expected = PreparedCakePlan(binned, q, 1.0).circular_fft()
    got = PreparedCakePlan(
        binned,
        q,
        1.0,
        harmonic_bandlimit_margin=10_000,
    ).circular_fft()

    assert relative_l2(got, expected) < 1e-12


def test_harmonic_bandlimit_low_q_approximates_exact_circular() -> None:
    rng = np.random.default_rng(16)
    coords = rng.normal(scale=0.5, size=(600, 3))
    q = np.linspace(0.05, 0.35, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=10,
        n_z=10,
        n_phi=128,
        r_max=2.0,
        z_range=(-2.0, 2.0),
    )

    expected_plan = PreparedCakePlan(binned, q, 1.0)
    got_plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        harmonic_bandlimit_margin=24,
    )
    assert got_plan._harmonic_indices_for_block(np.arange(q.size)) is not None

    expected = expected_plan.circular_fft()
    got = got_plan.circular_fft()

    assert relative_l2(got, expected) < 1e-10


def test_cpp_circular_backend_matches_numpy_with_harmonic_bandlimit() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(17)
    coords = rng.normal(scale=0.7, size=(700, 3))
    q = np.linspace(0.05, 0.4, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=11,
        n_z=9,
        n_phi=128,
        r_max=2.5,
        z_range=(-2.5, 2.5),
    )

    expected = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="numpy",
        harmonic_bandlimit_margin=24,
    ).circular_fft()
    got = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        harmonic_bandlimit_margin=24,
    ).circular_fft()

    assert relative_l2(got, expected) < 1e-12


def test_float32_histogram_fast_path_matches_default_circular() -> None:
    rng = np.random.default_rng(18)
    coords = rng.normal(size=(1000, 3))
    q = np.linspace(0.1, 1.4, 9)
    kwargs = {"n_r": 14, "n_z": 13, "n_phi": 72}

    default_binned = make_cylindrical_histogram(coords, **kwargs)
    float32_binned = make_cylindrical_histogram(
        coords,
        hist_dtype=np.float32,
        backend="cpp",
        **kwargs,
    )

    expected = PreparedCakePlan(
        default_binned,
        q,
        1.0,
        circular_backend="numpy",
    ).circular_fft()
    got = PreparedCakePlan(
        float32_binned,
        q,
        1.0,
        circular_backend="numpy",
    ).circular_fft()

    assert float32_binned.hist.dtype == np.float32
    assert relative_l2(got, expected) < 5e-6


def test_float32_histogram_auto_complex64_for_high_phi() -> None:
    rng = np.random.default_rng(181)
    coords = rng.normal(size=(1000, 3))
    q = np.linspace(0.1, 1.4, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=14,
        n_z=13,
        n_phi=720,
        hist_dtype=np.float32,
        backend="cpp",
    )

    plan = PreparedCakePlan(binned, q, 1.0, complex_dtype=None)
    reference = PreparedCakePlan(
        binned,
        q,
        1.0,
        complex_dtype=np.complex128,
    ).circular_fft()
    got = plan.circular_fft()

    assert plan.complex_dtype == np.dtype("complex64")
    assert relative_l2(got, reference) < 1e-6


def test_sparse_rz_circular_matches_dense_circular() -> None:
    rng = np.random.default_rng(19)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.4, 9)
    binned = make_cylindrical_histogram(coords, n_r=15, n_z=14, n_phi=80)
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy")

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_rz(active_chunk_size=17)

    assert plan.active_rz_count <= binned.hist.shape[0] * binned.hist.shape[1] * binned.hist.shape[2]
    assert relative_l2(got, expected) < 1e-12


def test_cpp_sparse_rz_circular_matches_dense_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(20)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.4, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=15,
        n_z=14,
        n_phi=80,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_rz()

    assert relative_l2(got, expected) < 5e-6


def test_cpp_sparse_flat_circular_matches_dense_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(21)
    coords = rng.normal(size=(700, 3))
    q = np.linspace(0.1, 1.4, 7)
    binned = make_cylindrical_histogram(
        coords,
        n_r=16,
        n_z=15,
        n_phi=64,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_flat()

    assert plan.active_flat_count <= binned.hist.size
    assert relative_l2(got, expected) < 5e-6


def test_cpp_sparse_flat_circular_matches_dense_with_bandlimit() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(22)
    coords = rng.normal(size=(700, 3))
    q = np.linspace(0.05, 0.5, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=16,
        n_z=15,
        n_phi=128,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        harmonic_bandlimit_margin=16,
    )

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_flat()

    assert relative_l2(got, expected) < 5e-6


def test_cpp_sparse_profiles_circular_matches_dense_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(23)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.3, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=17,
        n_z=16,
        n_phi=72,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_profiles()

    assert plan.active_sparse_profile_count <= plan.active_flat_count
    assert relative_l2(got, expected) < 5e-6


def test_cpp_sparse_profiles_circular_matches_dense_with_bandlimit() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(24)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.05, 0.45, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=17,
        n_z=16,
        n_phi=144,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        harmonic_bandlimit_margin=16,
    )

    expected = plan.circular_fft()
    got = plan.circular_fft_sparse_profiles()

    assert relative_l2(got, expected) < 5e-6


def test_sparse_source_projection_matches_dense_circular() -> None:
    rng = np.random.default_rng(241)
    coords = rng.normal(size=(600, 3))
    q = np.linspace(0.1, 1.5, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=22,
        n_phi=96,
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy", q_block_size=3)

    expected = plan.circular_fft(q_block_size=3)
    got = plan.circular_fft_sparse_source_projection(
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert plan.active_er_profile_count <= binned.hist.shape[0] * binned.hist.shape[1]
    assert relative_l2(got, expected) < 1e-12


def test_sparse_binned_structure_matches_dense_sparse_source() -> None:
    rng = np.random.default_rng(24101)
    coords = rng.normal(size=(700, 3))
    q = np.linspace(0.08, 0.85, 8)
    n_r, n_z, n_phi = 18, 20, 128
    r_max = float(np.sqrt(np.sum(coords[:, :2] ** 2, axis=1)).max()) * 1.01
    z_range = (float(coords[:, 2].min()) - 0.01, float(coords[:, 2].max()) + 0.01)
    dense = make_cylindrical_histogram(
        coords,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        hist_dtype=np.float32,
    )
    flat = cylindrical_flat_indices(
        coords,
        backend="numpy",
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
    )
    sparse = make_sparse_cylindrical_histogram_from_flat_indices(
        flat,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=r_max,
        z_range=z_range,
        element_order=("X",),
    )

    expected = PreparedCakePlan(
        dense,
        q,
        1.0,
        circular_backend="numpy",
        q_block_size=3,
    ).circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=True,
        q_block_size=3,
        profile_chunk_size=5,
    )
    sparse_plan = PreparedCakePlan(
        sparse,
        q,
        1.0,
        circular_backend="numpy",
        q_block_size=3,
    )
    got = sparse_plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=True,
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert sparse.active_values.size == np.count_nonzero(dense.hist)
    assert relative_l2(got, expected) < 1e-12
    with pytest.raises(RuntimeError, match="sparse-source solver"):
        sparse_plan.circular_fft()


def test_cpp_sparse_source_projection_matches_dense_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(2411)
    coords = rng.normal(size=(700, 3))
    q = np.linspace(0.1, 1.5, 9)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=22,
        n_phi=96,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        q_block_size=3,
    )

    expected = plan.circular_fft(q_block_size=3)
    got = plan.circular_fft_sparse_source_projection(
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert plan.active_er_profile_count <= binned.hist.shape[0] * binned.hist.shape[1]
    assert relative_l2(got, expected) < 5e-6


def test_sparse_source_projection_matches_dense_with_bandlimit() -> None:
    rng = np.random.default_rng(242)
    coords = rng.normal(size=(600, 3))
    q = np.linspace(0.05, 0.45, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=24,
        n_phi=160,
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        q_block_size=2,
        harmonic_bandlimit_margin=16,
    )

    expected = plan.circular_fft(q_block_size=2)
    got = plan.circular_fft_sparse_source_projection(
        q_block_size=2,
        profile_chunk_size=4,
    )

    assert relative_l2(got, expected) < 1e-12


def test_sparse_source_r_dependent_matches_r_dependent_cake() -> None:
    rng = np.random.default_rng(243)
    coords = rng.normal(size=(600, 3))
    q = np.linspace(0.08, 0.85, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=20,
        n_phi=128,
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy", q_block_size=3)

    expected = plan.circular_fft_r_dependent_bandlimit(
        margin=8,
        cutoff_bin_size=8,
        q_block_size=3,
    )
    got = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert relative_l2(got, expected) < 1e-12


def test_cpp_sparse_source_r_dependent_matches_r_dependent_cake() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(244)
    coords = rng.normal(size=(700, 3))
    q = np.linspace(0.08, 0.85, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=20,
        n_phi=128,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        q_block_size=3,
    )

    expected = plan.circular_fft_r_dependent_bandlimit(
        margin=8,
        cutoff_bin_size=8,
        q_block_size=3,
    )
    got = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert relative_l2(got, expected) < 5e-6


def test_sparse_source_r_dependent_analytic_kernel_matches_sampled() -> None:
    rng = np.random.default_rng(245)
    coords = rng.normal(size=(650, 3))
    q = np.linspace(0.08, 0.75, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=20,
        n_phi=128,
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy", q_block_size=3)

    expected = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=False,
        q_block_size=3,
        profile_chunk_size=5,
    )
    got = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=True,
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert relative_l2(got, expected) < 1e-10


def test_cpp_sparse_source_r_dependent_analytic_kernel_matches_sampled() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(246)
    coords = rng.normal(size=(750, 3))
    q = np.linspace(0.08, 0.75, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=18,
        n_z=20,
        n_phi=128,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        complex_dtype=np.complex64,
        q_block_size=3,
    )

    expected = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=False,
        q_block_size=3,
        profile_chunk_size=5,
    )
    got = plan.circular_fft_sparse_source_r_dependent(
        margin=8,
        cutoff_bin_size=8,
        analytic_kernel=True,
        q_block_size=3,
        profile_chunk_size=5,
    )

    assert relative_l2(got, expected) < 5e-6


def test_adaptive_profiles_dense_rows_match_dense_circular() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(25)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.3, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=17,
        n_z=16,
        n_phi=72,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")

    expected = plan.circular_fft()
    got = plan.circular_fft_adaptive_profiles(dense_row_factor=0.0)

    stats = plan.last_adaptive_profile_stats
    assert stats is not None
    assert stats["max_dense_profile_count"] == plan.active_sparse_profile_count
    assert relative_l2(got, expected) < 5e-6


def test_adaptive_profiles_dense_rows_match_dense_with_bandlimit() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(26)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.05, 0.45, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=17,
        n_z=16,
        n_phi=144,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(
        binned,
        q,
        1.0,
        circular_backend="cpp",
        harmonic_bandlimit_margin=16,
    )

    expected = plan.circular_fft()
    got = plan.circular_fft_adaptive_profiles(dense_row_factor=0.0)

    assert relative_l2(got, expected) < 5e-6


def test_sparse_profile_ring_average_matches_dense_curve() -> None:
    pytest.importorskip("waxs_cake._cpp_solvers")

    rng = np.random.default_rng(27)
    coords = rng.normal(size=(900, 3))
    q = np.linspace(0.1, 1.2, 8)
    binned = make_cylindrical_histogram(
        coords,
        n_r=16,
        n_z=15,
        n_phi=96,
        hist_dtype=np.float32,
        backend="cpp",
    )
    plan = PreparedCakePlan(binned, q, 1.0, circular_backend="cpp")

    expected = plan.ring_average_intensity()
    sparse = plan.ring_average_intensity_sparse_profiles()
    adaptive = plan.ring_average_intensity_adaptive_profiles(dense_row_factor=0.0)

    assert relative_l2(sparse, expected) < 5e-6
    assert relative_l2(adaptive, expected) < 5e-6
