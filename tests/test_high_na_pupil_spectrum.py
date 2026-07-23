from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_debye_wolf import (  # noqa: E402
    _union_h_positions,
    focal_axes,
    flatten_focal_axes,
    resolve_pupil_spectrum_required_h_abs,
)
from validate_high_na_harmonic_support_risk import evaluate_vector_stress  # noqa: E402


def _pupil_with_modes(nphi: int, modes: tuple[int, ...]) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    pupil = sum(np.exp(1j * mode * phi) for mode in modes)
    return np.asarray(pupil[None, :], dtype=np.complex128)


def test_adaptive_preserves_significant_modes_inside_global_cutoff() -> None:
    required = resolve_pupil_spectrum_required_h_abs(
        None,
        nphi=96,
        geometric_h_cutoff=17,
        pupil_spectrum="adaptive",
        pupil_spectrum_pupils=_pupil_with_modes(96, (16, 17, 18, 20)),
        relative_threshold=1e-8,
    )

    np.testing.assert_array_equal(required, np.array([16, 17, 18, 20]))
    np.testing.assert_array_equal(
        _union_h_positions(8, required, max_cutoff=20),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 20]),
    )


def test_warn_and_off_keep_manual_requirements_only() -> None:
    pupil = _pupil_with_modes(96, (16, 18))
    manual = np.array([3], dtype=np.int64)

    off = resolve_pupil_spectrum_required_h_abs(
        manual,
        nphi=96,
        geometric_h_cutoff=17,
        pupil_spectrum="off",
    )
    with pytest.warns(RuntimeWarning):
        warn = resolve_pupil_spectrum_required_h_abs(
            manual,
            nphi=96,
            geometric_h_cutoff=17,
            pupil_spectrum="warn",
            pupil_spectrum_pupils=pupil,
            relative_threshold=1e-8,
        )

    np.testing.assert_array_equal(off, manual)
    np.testing.assert_array_equal(warn, manual)


def test_charge18_effective_vector_adaptive_recovers_direct_reference() -> None:
    rho_axis, psi_axis, z_axis = focal_axes(
        nrho=6,
        npsi=8,
        nz=3,
        rho_max=2.0,
        z_max=0.5,
    )
    rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    result = evaluate_vector_stress(
        sin_theta_max=0.8,
        vortex_charge=18,
        ntheta=16,
        nphi=64,
        k=2.0 * np.pi,
        margin=6,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        rho=rho,
        psi=psi,
        z=z,
    )
    variants = result["variants"]

    assert variants["geometric_only"]["complex_l2"] > 0.1
    assert variants["adaptive_raw_jones"]["complex_l2"] > 1e-4
    assert variants["adaptive_effective_vector"]["complex_l2"] < 1e-6
    assert (
        variants["adaptive_effective_vector"]["required_h_abs"]
        == result["effective_vector_significant_h_abs"]
    )
