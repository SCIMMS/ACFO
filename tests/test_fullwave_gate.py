from __future__ import annotations

import numpy as np

from waxs_cake import summarize_fullwave_arrays


def test_fullwave_gate_uses_background_gain_and_passes_controlled_bundle() -> None:
    angles = np.linspace(-3.0, 3.0, 61)
    contrasts = np.array([1.0, 0.5, 0.25, 0.125])
    resolutions = np.array([12.0, 16.0, 20.0])
    envelope = np.exp(-0.5 * (angles / 0.9) ** 2)
    background = np.stack((envelope, 0.3 * envelope, 0.1j * envelope), axis=-1)
    acfo = np.stack([background * (1.0 + 0.05 * value) for value in contrasts])
    direct = acfo * (1.0 + 1e-6)
    fullwave = np.empty((resolutions.size, contrasts.size, angles.size, 3), dtype=np.complex128)
    forced = np.empty_like(fullwave)
    background_fullwave = np.empty((resolutions.size, angles.size, 3), dtype=np.complex128)
    for ir, resolution in enumerate(resolutions):
        scale = (1.7 + 0.2j) * (1.0 + 0.01 / resolution)
        background_fullwave[ir] = background / scale
        grid_error = 0.003 * (resolutions[-1] / resolution - 1.0)
        for ic, contrast in enumerate(contrasts):
            physics = 0.025 * contrast
            fullwave[ir, ic] = acfo[ic] * (1.0 + physics + grid_error) / scale
            shifted = np.roll(acfo[ic], 2, axis=0)
            forced[ir, ic] = (shifted + 0.4 * contrast * acfo[ic]) / scale
    result = summarize_fullwave_arrays(
        angles_deg=angles,
        contrast_scales=contrasts,
        resolutions=resolutions,
        background_born_field=background,
        background_fullwave_field=background_fullwave,
        acfo_field=acfo,
        direct_born_field=direct,
        fullwave_field=fullwave,
        forced_sphere_field=forced,
    )
    assert result["passed"]
    assert result["convergence"]["fullwave_born_loglog_slope"] > 0.9


def test_fullwave_gate_rejects_missing_contrast_convergence() -> None:
    angles = np.linspace(-1.0, 1.0, 9)
    contrasts = np.array([1.0, 0.5, 0.25])
    resolutions = np.array([8.0, 10.0, 12.0])
    background = np.ones((angles.size, 3), dtype=np.complex128)
    acfo = np.broadcast_to(background, (contrasts.size,) + background.shape).copy()
    fullwave = np.broadcast_to(1.1 * acfo, (resolutions.size,) + acfo.shape).copy()
    result = summarize_fullwave_arrays(
        angles_deg=angles,
        contrast_scales=contrasts,
        resolutions=resolutions,
        background_born_field=background,
        background_fullwave_field=np.broadcast_to(background, (resolutions.size,) + background.shape),
        acfo_field=acfo,
        direct_born_field=acfo,
        fullwave_field=fullwave,
        forced_sphere_field=1.3 * fullwave,
    )
    assert not result["gates"]["contrast_to_zero_converges"]
    assert not result["passed"]
