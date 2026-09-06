"""Independent checks for the validation-only disk calculus."""
import numpy as np
import pytest

from scripts.acfo_elliptic_disk import (
    DiskSpace, apply_physical_operator, assemble, gradient, load_vector,
    manufactured_fields, physical_errors, radial_rule, state_values, zernike_values,
    direct_solve,
)


@pytest.mark.parametrize("m", [-19, -3, 0, 2, 18])
def test_gradient_matches_analytic_radial_derivatives(m):
    space = DiskSpace((m,), 9)
    r, _ = radial_rule(48)
    for sign in [-1, 1]:
        g, labels = gradient(space, sign)
        z = np.column_stack([zernike_values(mm, [n], r)[:, 0] for mm, n in labels])
        for n in (0, 3, 8):
            u, ur, _ = state_values(m, n, r)
            expected = (ur-sign*m*u/r)/np.sqrt(2)
            np.testing.assert_allclose(z@g[:, n].toarray().ravel(), expected,
                                       rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("modes", [tuple(range(-5, 6)), (-19, -12, -3, 0, 3, 19, 26)])
def test_sparse_operator_against_independent_two_dimensional_weak_form(modes):
    space = DiskSpace(modes, 4)
    op = assemble(space, {3: 0.6, 7: 0.4}, quadrature=70)
    r, w = radial_rule(95)
    phi = 2*np.pi*(np.arange(128)+0.317)/128
    rng = np.random.default_rng(8542)
    x = rng.normal(size=space.size)+1j*rng.normal(size=space.size)
    y = rng.normal(size=space.size)+1j*rng.normal(size=space.size)
    def synthesize(c):
        u = np.zeros((r.size, phi.size), dtype=complex)
        ur = np.zeros_like(u); up = np.zeros_like(u)
        for j, (m, n) in enumerate(space.labels):
            b, br, _ = state_values(m, n, r)
            phase = np.exp(1j*m*phi)/np.sqrt(2*np.pi)
            u += c[j]*b[:, None]*phase
            ur += c[j]*br[:, None]*phase
            up += c[j]*(1j*m*b/r)[:, None]*phase
        return u, ur, up
    xu, xr, xp = synthesize(x)
    yu, yr, yp = synthesize(y)
    epsilon = 0.7
    a = 1+0.2*r[:, None]**2+epsilon*(0.6*r[:, None]**3*np.cos(3*phi)
                                          +0.4*r[:, None]**7*np.cos(7*phi))
    integrand = a*(xr.conj()*yr+xp.conj()*yp)+0.3*xu.conj()*yu
    reference = np.sum(w[:, None]*integrand)*2*np.pi/phi.size
    actual = np.vdot(x, op.matrix(epsilon)@y)
    assert abs(actual-reference)/abs(reference) < 2e-12


def test_nonpolynomial_manufactured_solution_and_strong_residual():
    space = DiskSpace(tuple(range(-20, 21)), 18)
    harmonics = {7: 1.0}
    epsilon = 0.4
    op = assemble(space, harmonics, quadrature=95)
    truth = lambda r: manufactured_fields(r, (0, 11))
    rhs = lambda r: apply_physical_operator(truth(r), r, harmonics, epsilon)
    f = load_vector(space, rhs, quadrature=120)
    u = direct_solve(op.matrix(epsilon), f)
    error = physical_errors(space, u, truth, harmonics=harmonics, epsilon=epsilon,
                            rhs=rhs, quadrature=155)
    assert error["physical_l2"] < 1e-10
    assert error["h1_seminorm"] < 1e-10
    assert error["strong_residual_l2"] < 1e-8


def test_exact_reachable_gradient_space_equals_full_padding():
    space = DiskSpace((-13, -1, 0, 1, 13), 7)
    good = assemble(space, {1: 0.4, 12: 0.6}, quadrature=75)
    padded = assemble(space, {1: 0.4, 12: 0.6}, quadrature=75, workspace="padded")
    np.testing.assert_allclose(good.matrix(0.5).toarray(), padded.matrix(0.5).toarray(),
                               atol=2e-13, rtol=2e-13)
    assert good.gradient_rows < padded.gradient_rows
