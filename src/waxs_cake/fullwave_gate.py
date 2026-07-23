"""Metric contract for an independent uniaxial Maxwell/FDTD validation run."""

from __future__ import annotations

from typing import Any

import numpy as np


def _relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(model.ravel()) == 0.0 else float("inf")
    return float(np.linalg.norm((model - reference).ravel()) / denominator)


def _intensity(field: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(field) ** 2, axis=-1)


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(x, y) / denominator)


def summarize_fullwave_arrays(
    *,
    angles_deg: np.ndarray,
    contrast_scales: np.ndarray,
    resolutions: np.ndarray,
    background_born_field: np.ndarray,
    background_fullwave_field: np.ndarray,
    acfo_field: np.ndarray,
    direct_born_field: np.ndarray,
    fullwave_field: np.ndarray,
    forced_sphere_field: np.ndarray,
) -> dict[str, Any]:
    """Calibrate, compare, and gate one full-wave result bundle.

    Full-wave fields are calibrated once per grid resolution against the
    zero-contrast background run.  The same gain is then used for every
    contrast and for the forced-sphere control; no per-case gain fitting is
    allowed.
    """

    angles = np.asarray(angles_deg, dtype=np.float64)
    contrasts = np.asarray(contrast_scales, dtype=np.float64)
    grids = np.asarray(resolutions, dtype=np.float64)
    background_born = np.asarray(background_born_field, dtype=np.complex128)
    background_fullwave = np.asarray(background_fullwave_field, dtype=np.complex128)
    acfo = np.asarray(acfo_field, dtype=np.complex128)
    direct = np.asarray(direct_born_field, dtype=np.complex128)
    fullwave = np.asarray(fullwave_field, dtype=np.complex128)
    forced = np.asarray(forced_sphere_field, dtype=np.complex128)
    a = angles.size
    c = contrasts.size
    r = grids.size
    expected_vector = (a, 3)
    expected_contrast = (c, a, 3)
    expected_full = (r, c, a, 3)
    if angles.ndim != 1 or contrasts.ndim != 1 or grids.ndim != 1:
        raise ValueError("angles, contrast_scales, and resolutions must be vectors")
    if a < 3 or c < 3 or r < 3:
        raise ValueError("at least 3 angles, contrasts, and resolutions are required")
    if np.any(np.diff(angles) <= 0.0) or np.any(contrasts <= 0.0) or np.any(grids <= 0.0):
        raise ValueError("angles must increase and contrasts/resolutions must be positive")
    if background_born.shape != expected_vector:
        raise ValueError(f"background_born_field must have shape {expected_vector}")
    if background_fullwave.shape != (r, a, 3):
        raise ValueError(f"background_fullwave_field must have shape {(r, a, 3)}")
    if acfo.shape != expected_contrast or direct.shape != expected_contrast:
        raise ValueError(f"Born fields must have shape {expected_contrast}")
    if fullwave.shape != expected_full or forced.shape != expected_full:
        raise ValueError(f"full-wave fields must have shape {expected_full}")
    arrays = (angles, contrasts, grids, background_born, background_fullwave, acfo, direct, fullwave, forced)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all full-wave gate arrays must be finite")

    gains = np.empty(r, dtype=np.complex128)
    calibrated = np.empty_like(fullwave)
    calibrated_forced = np.empty_like(forced)
    background_errors: list[float] = []
    for ir in range(r):
        denominator = np.vdot(background_fullwave[ir], background_fullwave[ir])
        if denominator == 0.0:
            raise ValueError("background_fullwave_field must be nonzero")
        gains[ir] = np.vdot(background_fullwave[ir], background_born) / denominator
        calibrated[ir] = gains[ir] * fullwave[ir]
        calibrated_forced[ir] = gains[ir] * forced[ir]
        background_errors.append(_relative_l2(gains[ir] * background_fullwave[ir], background_born))

    finest = int(np.argmax(grids))
    next_finest = int(np.argsort(grids)[-2])
    nominal = int(np.argmax(contrasts))
    weakest = int(np.argmin(contrasts))
    angle_step = float(np.median(np.diff(angles)))
    per_contrast: list[dict[str, float]] = []
    for ic, contrast in enumerate(contrasts):
        correct = calibrated[finest, ic]
        wrong = calibrated_forced[finest, ic]
        born = acfo[ic]
        direct_case = direct[ic]
        discrepancy = _relative_l2(correct, born)
        algorithm_error = _relative_l2(born, direct_case)
        correct_intensity = _intensity(correct)
        born_intensity = _intensity(born)
        wrong_intensity = _intensity(wrong)
        correct_peak = float(angles[int(np.argmax(correct_intensity))])
        born_peak = float(angles[int(np.argmax(born_intensity))])
        wrong_peak = float(angles[int(np.argmax(wrong_intensity))])
        correct_residual = float(np.linalg.norm((correct - born).ravel()))
        wrong_residual = float(np.linalg.norm((wrong - correct).ravel()))
        per_contrast.append(
            {
                "contrast_scale": float(contrast),
                "fullwave_vs_acfo_complex_l2": discrepancy,
                "acfo_vs_direct_complex_l2": algorithm_error,
                "algorithm_to_physics_error_ratio": algorithm_error / discrepancy if discrepancy > 0.0 else float("inf"),
                "intensity_ncc": _ncc(correct_intensity, born_intensity),
                "peak_angle_error_deg": abs(correct_peak - born_peak),
                "forced_sphere_wrong_to_correct_residual_ratio": wrong_residual / correct_residual if correct_residual > 0.0 else float("inf"),
                "forced_sphere_peak_shift_deg": abs(wrong_peak - correct_peak),
                "forced_sphere_peak_shift_bins": abs(wrong_peak - correct_peak) / angle_step,
                "forced_sphere_intensity_correlation_drop": 1.0 - _ncc(wrong_intensity, correct_intensity),
            }
        )

    order = np.argsort(contrasts)
    sorted_contrast = contrasts[order]
    sorted_error = np.array([per_contrast[index]["fullwave_vs_acfo_complex_l2"] for index in order])
    positive = (sorted_contrast > 0.0) & (sorted_error > 0.0)
    if np.count_nonzero(positive) >= 2:
        convergence_slope = float(np.polyfit(np.log(sorted_contrast[positive]), np.log(sorted_error[positive]), 1)[0])
    else:
        convergence_slope = float("nan")
    monotonic_violations = int(np.count_nonzero(np.diff(sorted_error) < -0.05 * np.maximum(sorted_error[:-1], 1e-30)))
    grid_convergence = _relative_l2(calibrated[finest, nominal], calibrated[next_finest, nominal])
    nominal_metrics = per_contrast[nominal]
    weakest_metrics = per_contrast[weakest]
    gates = {
        "background_calibration_l2_max_le_2pct": max(background_errors) <= 0.02,
        "weakest_contrast_complex_l2_le_5pct": weakest_metrics["fullwave_vs_acfo_complex_l2"] <= 0.05,
        "contrast_to_zero_converges": bool(np.isfinite(convergence_slope) and convergence_slope >= 0.5 and monotonic_violations == 0),
        "nominal_intensity_ncc_ge_0_98": nominal_metrics["intensity_ncc"] >= 0.98,
        "nominal_peak_error_le_0_2deg": nominal_metrics["peak_angle_error_deg"] <= 0.2,
        "forced_sphere_ratio_ge_5": nominal_metrics["forced_sphere_wrong_to_correct_residual_ratio"] >= 5.0,
        "forced_sphere_observable": (
            nominal_metrics["forced_sphere_peak_shift_bins"] >= 1.0
            or nominal_metrics["forced_sphere_intensity_correlation_drop"] >= 0.10
        ),
        "algorithm_error_lt_1pct_physics_mismatch": all(
            item["algorithm_to_physics_error_ratio"] < 0.01 for item in per_contrast
        ),
        "three_level_grid_convergence_le_2pct": grid_convergence <= 0.02,
    }
    return {
        "calibration": {
            "complex_gains": [[float(value.real), float(value.imag)] for value in gains],
            "background_relative_l2_by_resolution": background_errors,
        },
        "per_contrast": per_contrast,
        "convergence": {
            "fullwave_born_loglog_slope": convergence_slope,
            "monotonic_violations": monotonic_violations,
            "nominal_next_to_finest_grid_complex_l2": grid_convergence,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
