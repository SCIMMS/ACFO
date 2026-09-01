"""Exact object-level factorization for the NuMagSANS Example 3 ensemble.

The public example uses two different ensemble contractions.  The dilute case
adds per-particle cross sections, whereas explicit structure tables translate
and coherently add particle amplitudes before the polarization contraction.
This module keeps that distinction explicit and supplies small NumPy oracles
for the prospective GPU benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


PRIMITIVE_OUTPUT_NAMES = ("S_N", "S_M", "S_NM", "S_P", "S_chi")
OUTPUT_NAMES = (
    "S_N",
    "S_M",
    "S_NM",
    "S_P",
    "S_chi",
    "S_sf",
    "S_pm",
    "S_mp",
    "S_pp",
    "S_mm",
    "S_p",
    "S_m",
)


def relative_l2(actual: Any, reference: Any) -> float:
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    return float(
        np.linalg.norm(actual_array - reference_array)
        / max(float(np.linalg.norm(reference_array)), 1e-30)
    )


def rotate_points_about_z(points: Any, angle: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    result = np.array(values, copy=True)
    result[:, 0] = cosine * values[:, 0] - sine * values[:, 1]
    result[:, 1] = sine * values[:, 0] + cosine * values[:, 1]
    return np.ascontiguousarray(result)


def load_structure_positions(
    path: str | Path,
    *,
    coordinate_scale: float = 1e9,
    rotate_beam_x_to_z: bool = True,
) -> np.ndarray:
    """Load and coordinate-match a NuMagSANS structure table."""

    positions = np.loadtxt(Path(path), dtype=np.float64)
    if positions.ndim == 1:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError("structure table must contain at least three columns")
    positions = positions[:, :3] * float(coordinate_scale)
    if rotate_beam_x_to_z:
        positions = positions[:, np.asarray([1, 2, 0], dtype=np.intp)]
    if not np.all(np.isfinite(positions)):
        raise ValueError("structure table contains non-finite values")
    return np.ascontiguousarray(positions)


def translation_phases(q_xyz: Any, centers: Any, *, phase_sign: int = -1) -> np.ndarray:
    """Return exact Fourier-translation phases for object centers."""

    q = np.asarray(q_xyz, dtype=np.float64)
    center_array = np.asarray(centers, dtype=np.float64)
    if q.ndim < 2 or q.shape[-1] != 3:
        raise ValueError("q_xyz must have shape (..., 3)")
    if center_array.ndim != 2 or center_array.shape[1] != 3:
        raise ValueError("centers must have shape (K, 3)")
    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be -1 or +1")
    phase = np.einsum("...d,kd->k...", q, center_array, optimize=True)
    return np.exp(1j * int(phase_sign) * phase)


def _normalize_polarization(polarization: Any) -> np.ndarray:
    p = np.asarray(polarization, dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("polarization must be a finite three-vector")
    norm = float(np.linalg.norm(p))
    if norm <= 0.0:
        raise ValueError("polarization must be nonzero")
    return p / norm


def primitive_observables_from_channels(
    channel_amplitudes: Any,
    q_hat: Any,
    polarization: Any,
) -> dict[str, np.ndarray]:
    """Contract four complex amplitudes into five independently scaled terms."""

    channels = np.asarray(channel_amplitudes)
    directions = np.asarray(q_hat, dtype=np.float64)
    if channels.ndim < 2 or channels.shape[0] != 4:
        raise ValueError("channel_amplitudes must have shape (4, ...)")
    if directions.shape != channels.shape[1:] + (3,):
        raise ValueError("q_hat must have shape channel_amplitudes.shape[1:] + (3,)")
    norms = np.linalg.norm(directions, axis=-1)
    if np.any(norms <= 0.0):
        raise ValueError("q_hat directions must be nonzero")
    directions = directions / norms[..., None]
    p = _normalize_polarization(polarization)
    nuclear_amp = channels[0]
    magnetization = np.moveaxis(channels[1:], 0, -1)
    interaction = directions * np.sum(
        directions * magnetization, axis=-1, keepdims=True
    ) - magnetization
    projected_amp = np.einsum("...i,i->...", interaction, p, optimize=True)
    nuclear = np.abs(nuclear_amp) ** 2
    magnetic = np.sum(np.abs(interaction) ** 2, axis=-1)
    projected = np.abs(projected_amp) ** 2
    nuclear_magnetic = 2.0 * np.real(nuclear_amp * np.conj(projected_amp))
    chiral = np.real(
        -1j
        * np.einsum(
            "i,...i->...",
            p,
            np.cross(interaction, np.conj(interaction)),
            optimize=True,
        )
    )
    return {
        "S_N": np.ascontiguousarray(nuclear),
        "S_M": np.ascontiguousarray(magnetic),
        "S_NM": np.ascontiguousarray(nuclear_magnetic),
        "S_P": np.ascontiguousarray(projected),
        "S_chi": np.ascontiguousarray(chiral),
    }


def derive_twelve_observables(
    primitive: Mapping[str, Any],
    *,
    nuclear_scale: float = 1.0,
    magnetic_scale: float = 1.0,
    nuclear_magnetic_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply NuMagSANS physical scale classes and derive all twelve outputs."""

    result = {
        "S_N": float(nuclear_scale) * np.asarray(primitive["S_N"]),
        "S_M": float(magnetic_scale) * np.asarray(primitive["S_M"]),
        "S_NM": float(nuclear_magnetic_scale) * np.asarray(primitive["S_NM"]),
        "S_P": float(magnetic_scale) * np.asarray(primitive["S_P"]),
        "S_chi": float(magnetic_scale) * np.asarray(primitive["S_chi"]),
    }
    result["S_sf"] = result["S_M"] - result["S_P"]
    result["S_pm"] = result["S_sf"] + result["S_chi"]
    result["S_mp"] = result["S_sf"] - result["S_chi"]
    result["S_pp"] = result["S_N"] + result["S_NM"] + result["S_P"]
    result["S_mm"] = result["S_N"] - result["S_NM"] + result["S_P"]
    result["S_p"] = result["S_pp"] + result["S_pm"]
    result["S_m"] = result["S_mm"] + result["S_mp"]
    return {name: np.ascontiguousarray(result[name]) for name in OUTPUT_NAMES}


def aggregate_hierarchy(
    orientation_channels: Any,
    q_xyz: Any,
    q_hat: Any,
    polarization: Any,
    packed_centers: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Small-memory NumPy oracle for dilute and packed ensemble algebra."""

    channels = np.asarray(orientation_channels)
    if channels.ndim < 3 or channels.shape[1] != 4:
        raise ValueError("orientation_channels must have shape (K, 4, ...)")
    primitive_sum: dict[str, np.ndarray] | None = None
    for orientation in channels:
        primitive = primitive_observables_from_channels(
            orientation, q_hat, polarization
        )
        if primitive_sum is None:
            primitive_sum = {
                name: np.array(value, copy=True) for name, value in primitive.items()
            }
        else:
            for name in PRIMITIVE_OUTPUT_NAMES:
                primitive_sum[name] += primitive[name]
    if primitive_sum is None:
        raise ValueError("at least one orientation is required")
    result = {"dilute": derive_twelve_observables(primitive_sum)}
    for name, centers in packed_centers.items():
        center_array = np.asarray(centers, dtype=np.float64)
        if center_array.shape != (channels.shape[0], 3):
            raise ValueError(f"packed centers for {name} must have shape (K, 3)")
        phases = translation_phases(q_xyz, center_array)
        coherent = np.sum(channels * phases[:, None, ...], axis=0)
        result[name] = derive_twelve_observables(
            primitive_observables_from_channels(coherent, q_hat, polarization)
        )
    return result


def first_strict_favorable_integer(
    candidate_total: Any, baseline_total: Any
) -> int | None:
    candidate = np.asarray(candidate_total, dtype=np.float64)
    baseline = np.asarray(baseline_total, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate_total and baseline_total must be matching vectors")
    favorable = np.flatnonzero(candidate < baseline)
    return None if favorable.size == 0 else int(favorable[0] + 1)


def first_persistent_favorable_integer(
    candidate_total: Any, baseline_total: Any
) -> int | None:
    candidate = np.asarray(candidate_total, dtype=np.float64)
    baseline = np.asarray(baseline_total, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate_total and baseline_total must be matching vectors")
    favorable = candidate < baseline
    persistent = np.logical_and.accumulate(favorable[::-1])[::-1]
    indices = np.flatnonzero(persistent)
    return None if indices.size == 0 else int(indices[0] + 1)


def cumulative_crossover(
    candidate_samples: Any,
    baseline_samples: Any,
    *,
    candidate_setup: float,
    baseline_setup: float,
    candidate_cold_start: float = 0.0,
) -> dict[str, Any]:
    """Evaluate measured warm and cold setup-inclusive orientation crossovers."""

    candidate = np.asarray(candidate_samples, dtype=np.float64)
    baseline = np.asarray(baseline_samples, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate_samples and baseline_samples must be matching vectors")
    warm_candidate = float(candidate_setup) + np.cumsum(candidate)
    cold_candidate = float(candidate_setup + candidate_cold_start) + np.cumsum(candidate)
    baseline_total = float(baseline_setup) + np.cumsum(baseline)
    return {
        "warm_first_strict": first_strict_favorable_integer(
            warm_candidate, baseline_total
        ),
        "warm_first_persistent": first_persistent_favorable_integer(
            warm_candidate, baseline_total
        ),
        "cold_first_strict": first_strict_favorable_integer(
            cold_candidate, baseline_total
        ),
        "cold_first_persistent": first_persistent_favorable_integer(
            cold_candidate, baseline_total
        ),
        "warm_candidate_total": warm_candidate,
        "cold_candidate_total": cold_candidate,
        "baseline_total": baseline_total,
    }


def frozen_dispatch_decision(
    orientation_count: int,
    *,
    qualification_pass: bool,
    cold_process: bool = True,
    warm_threshold: int = 37,
    cold_threshold: int = 50,
) -> str:
    if int(orientation_count) <= 0:
        raise ValueError("orientation_count must be positive")
    if not qualification_pass:
        return "PROJECTED_TYPE3"
    threshold = int(cold_threshold if cold_process else warm_threshold)
    return "ACFO" if int(orientation_count) >= threshold else "PROJECTED_TYPE3"


def streaming_state_bytes(
    *,
    targets: int,
    packed_cases: int = 4,
    channels: int = 4,
    primitive_outputs: int = 5,
    complex_bytes: int = 8,
    real_bytes: int = 4,
    method_arms: int = 2,
) -> int:
    """Upper bound for two streamed method arms without an amplitude library."""

    per_arm = (
        int(packed_cases) * int(channels) * int(targets) * int(complex_bytes)
        + int(primitive_outputs) * int(targets) * int(real_bytes)
    )
    return int(method_arms) * per_arm


def certify_affine_cartesian_lattice(
    points: Any,
    *,
    basis_anchor_rows: tuple[int, int, int] = (1, 5, 49),
    expected_shape: tuple[int, int, int] | None = (20, 20, 20),
    coordinate_tolerance: float = 1e-8,
    orthogonality_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Certify a rotated/translated Cartesian lattice without axis alignment.

    NuMagSANS Example 3 stores the same rigid lattice in every orientation.
    Rows 1, 5, and 49 are the three one-cell neighbours of row 0 in the
    public archive.  The resulting affine representation is

    ``r[j] = origin + basis @ indices[j]``.

    The integer coordinates, full dense shape, uniqueness, reconstruction
    residual, and orthogonality are all checked.  The returned centered origin
    is the phase origin required by a mode-ordered type-2 NUFFT.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if values.shape[0] <= max(basis_anchor_rows):
        raise ValueError("basis anchor row is outside the source table")
    if not np.all(np.isfinite(values)):
        raise ValueError("points contain non-finite coordinates")
    if coordinate_tolerance <= 0.0 or orthogonality_tolerance <= 0.0:
        raise ValueError("lattice tolerances must be positive")

    anchor = values[0]
    basis = np.column_stack(
        [values[index] - anchor for index in basis_anchor_rows]
    )
    determinant = float(np.linalg.det(basis))
    if abs(determinant) <= np.finfo(np.float64).eps:
        raise ValueError("basis anchors are singular")
    gram = basis.T @ basis
    spacing = np.sqrt(np.diag(gram))
    normalized_gram = gram / np.outer(spacing, spacing)
    maximum_off_diagonal = float(
        np.max(np.abs(normalized_gram - np.eye(3, dtype=np.float64)))
    )
    if maximum_off_diagonal > float(orthogonality_tolerance):
        raise ValueError(
            "basis anchors do not define an orthogonal Cartesian lattice: "
            f"max normalized Gram residual={maximum_off_diagonal}"
        )

    fractional = np.linalg.solve(basis, (values - anchor).T).T
    integer = np.rint(fractional).astype(np.int64)
    maximum_integer_residual = float(np.max(np.abs(fractional - integer)))
    lower_integer = np.min(integer, axis=0)
    upper_integer = np.max(integer, axis=0)
    shape = upper_integer - lower_integer + 1
    if expected_shape is not None and tuple(int(v) for v in shape) != tuple(
        int(v) for v in expected_shape
    ):
        raise ValueError(
            "affine lattice shape differs from the frozen contract: "
            f"observed={tuple(int(v) for v in shape)}, expected={expected_shape}"
        )
    indices = integer - lower_integer
    if np.unique(indices, axis=0).shape[0] != values.shape[0]:
        raise ValueError("multiple active sites map to the same lattice cell")
    if np.any(indices < 0) or np.any(indices >= shape[None, :]):
        raise ValueError("affine lattice indices escaped the inferred shape")

    origin = anchor + basis @ lower_integer
    reconstructed = origin[None, :] + indices @ basis.T
    maximum_coordinate_residual = float(
        np.max(np.abs(reconstructed - values))
    )
    if maximum_coordinate_residual > float(coordinate_tolerance):
        raise ValueError(
            "source coordinates fail the affine Cartesian reconstruction: "
            f"residual={maximum_coordinate_residual}"
        )
    center_index = shape // 2
    center = origin + basis @ center_index
    return {
        "anchor_row": 0,
        "basis_anchor_rows": tuple(int(v) for v in basis_anchor_rows),
        "basis": np.ascontiguousarray(basis),
        "gram": np.ascontiguousarray(gram),
        "spacing": np.ascontiguousarray(spacing),
        "determinant": determinant,
        "integer_lower": np.ascontiguousarray(lower_integer),
        "integer_upper": np.ascontiguousarray(upper_integer),
        "shape": np.ascontiguousarray(shape),
        "indices": np.ascontiguousarray(indices),
        "origin": np.ascontiguousarray(origin),
        "center_index": np.ascontiguousarray(center_index),
        "center": np.ascontiguousarray(center),
        "active_sites": int(values.shape[0]),
        "dense_sites": int(np.prod(shape, dtype=np.int64)),
        "maximum_integer_residual": maximum_integer_residual,
        "maximum_coordinate_residual": maximum_coordinate_residual,
        "maximum_normalized_gram_residual": maximum_off_diagonal,
    }


def compare_affine_lattice_certificates(
    actual: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    gram_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check that two affine certificates differ only by a rigid rotation."""

    actual_shape = np.asarray(actual["shape"], dtype=np.int64)
    reference_shape = np.asarray(reference["shape"], dtype=np.int64)
    shape_equal = bool(np.array_equal(actual_shape, reference_shape))
    indices_equal = bool(
        np.array_equal(
            np.asarray(actual["indices"], dtype=np.int64),
            np.asarray(reference["indices"], dtype=np.int64),
        )
    )
    gram_actual = np.asarray(actual["gram"], dtype=np.float64)
    gram_reference = np.asarray(reference["gram"], dtype=np.float64)
    gram_max_abs = float(np.max(np.abs(gram_actual - gram_reference)))
    basis_actual = np.asarray(actual["basis"], dtype=np.float64)
    basis_reference = np.asarray(reference["basis"], dtype=np.float64)
    rigid_map = basis_actual @ np.linalg.inv(basis_reference)
    rigid_residual = float(
        np.max(np.abs(rigid_map.T @ rigid_map - np.eye(3)))
    )
    rigid_determinant = float(np.linalg.det(rigid_map))
    passed = bool(
        shape_equal
        and indices_equal
        and gram_max_abs <= float(gram_tolerance)
        and rigid_residual <= float(gram_tolerance)
        and rigid_determinant > 0.0
    )
    return {
        "pass": passed,
        "shape_equal": shape_equal,
        "indices_equal": indices_equal,
        "gram_max_abs": gram_max_abs,
        "rigid_map_orthogonality_max_abs": rigid_residual,
        "rigid_map_determinant": rigid_determinant,
    }


def affine_lattice_channels(
    certificate: Mapping[str, Any],
    nuclear: Any,
    magnetization: Any,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Scatter four active-site channels into the certified dense lattice."""

    indices = np.asarray(certificate["indices"], dtype=np.int64)
    shape = tuple(int(v) for v in np.asarray(certificate["shape"]))
    nuclear_values = np.asarray(nuclear)
    magnetic_values = np.asarray(magnetization)
    if nuclear_values.shape != (indices.shape[0],):
        raise ValueError("nuclear weights do not match the certified sites")
    if magnetic_values.shape != (indices.shape[0], 3):
        raise ValueError("magnetization must have shape (N, 3)")
    channels = np.zeros((4, *shape), dtype=dtype)
    location = tuple(indices[:, axis] for axis in range(3))
    channels[(0, *location)] = nuclear_values
    for channel in range(3):
        channels[(channel + 1, *location)] = magnetic_values[:, channel]
    return np.ascontiguousarray(channels)


def wrap_type2_targets(values: Any) -> np.ndarray:
    """Wrap dimensionless type-2 targets to cuFINUFFT's principal interval."""

    targets = np.asarray(values, dtype=np.float64)
    wrapped = np.remainder(targets + np.pi, 2.0 * np.pi) - np.pi
    return np.ascontiguousarray(wrapped)


def affine_lattice_type2_contract(
    certificate: Mapping[str, Any], q_xyz: Any
) -> dict[str, np.ndarray]:
    """Return exact centered type-2 targets and physical phase correction."""

    q = np.asarray(q_xyz, dtype=np.float64)
    if q.ndim < 2 or q.shape[-1] != 3:
        raise ValueError("q_xyz must have shape (..., 3)")
    flat = q.reshape(-1, 3)
    basis = np.asarray(certificate["basis"], dtype=np.float64)
    center = np.asarray(certificate["center"], dtype=np.float64)
    unwrapped = flat @ basis
    return {
        "scaled_targets_unwrapped": np.ascontiguousarray(unwrapped),
        "scaled_targets": wrap_type2_targets(unwrapped),
        "center_phase": np.ascontiguousarray(np.exp(-1j * (flat @ center))),
    }
