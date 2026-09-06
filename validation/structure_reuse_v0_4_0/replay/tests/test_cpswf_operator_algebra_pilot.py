"""Independent low-degree polynomial and angular ladder checks."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("algebra_pilot", Path(__file__).resolve().parents[1]/"scripts/experiment_cpswf_operator_algebra.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_jacobi_degree_one_ladder_against_explicit_polynomial():
    x = np.linspace(0.03, 0.97, 80)
    for m in (0, 1, 4, 12):
        scale = np.sqrt(2*(m+3))
        # physical R_1 = scale*r^m*((m+2)*r^2-(m+1)).
        raised = 2*scale*(m+2)*x**(m+1.5)
        lap = 4*scale*(m+1)*(m+2)*x**(m+0.5)
        np.testing.assert_allclose(PILOT.differential_columns(m, 2, x, "raise")[:, 1], raised, rtol=2e-14)
        np.testing.assert_allclose(PILOT.differential_columns(m, 2, x, "laplacian")[:, 1], lap, rtol=2e-14)
        np.testing.assert_array_equal(PILOT.differential_columns(m, 2, x, "raise")[:, 0], 0)


def test_cartesian_ladder_angular_selection_and_return():
    phi = 2*np.pi*np.arange(64)/64
    r, m, power = 0.7, 4, 8
    f = r**power*np.exp(1j*m*phi)
    dr = power/r*f
    dphi = 1j*m*f
    dx = np.cos(phi)*dr-np.sin(phi)/r*dphi
    dy = np.sin(phi)*dr+np.cos(phi)/r*dphi
    expected = (power-m)*r**(power-1)*np.exp(1j*(m+1)*phi)
    np.testing.assert_allclose(dx+1j*dy, expected, atol=2e-14)
    spectrum = np.fft.fft(dx+1j*dy)/len(phi)
    assert abs(spectrum[m+1]) > 0.1
    spectrum[m+1] = 0
    assert np.max(abs(spectrum)) < 2e-14


def test_nested_observation_scan_does_not_assume_monotonicity():
    assert PILOT.first_pass([1, 0.01, 0.5, 0.001], 0.02) == 1
    assert PILOT.first_pass([1, 0.1], 0.01) is None


def test_algebraic_ladder_and_suffix_sums_against_pointwise_derivatives():
    x, w = PILOT.accurate_gauss(100)
    rng = np.random.default_rng(910)
    for m in (0, 4, 12):
        n = 24
        dp, dm = PILOT.ladder_matrices(m, n)
        p = PILOT.radial_jacobi_basis(m, n, x)
        pp = PILOT.radial_jacobi_basis(m+1, n, x)
        assert PILOT.relative(pp@dp, PILOT.differential_columns(m, n, x, 'raise')) < 1e-12
        assert PILOT.relative(p@dm, PILOT.differential_columns(m+1, n, x, 'lower')) < 1e-12
        z = rng.normal(size=(n, 4))+1j*rng.normal(size=(n, 4))
        np.testing.assert_allclose(PILOT.ladder_apply(m, z), dp@z, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(PILOT.ladder_apply(m, z, lower=True), dm@z, rtol=1e-13, atol=1e-12)
