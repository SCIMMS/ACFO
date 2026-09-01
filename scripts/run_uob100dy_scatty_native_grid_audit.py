#!/usr/bin/env python3
"""Isolate the UoB-100(Dy) SCATTY native-grid effect.

The completed 23 x 4 x 80 spin ensemble is reused byte-for-byte.  No Monte
Carlo work is performed here.  The primary maps use the 24 x 24 supercell
Bragg grid (1/24 r.l.u.) disclosed in the SI, with no Lanczos resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_base_runner() -> Any:
    path = Path(__file__).with_name("run_uob100dy_atomic_decoration_validation.py")
    spec = importlib.util.spec_from_file_location("uob100dy_atomic_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def native_axis(start: float, count: int, denominator: int = 24) -> np.ndarray:
    return float(start) + np.arange(int(count), dtype=np.float64) / float(denominator)


def _brackets(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source.ndim != 1 or np.any(np.diff(source) <= 0.0):
        raise ValueError("source axis must be strictly increasing")
    if np.min(target) < source[0] - 1e-12 or np.max(target) > source[-1] + 1e-12:
        raise ValueError("target axis lies outside source axis")
    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, source.size - 1)
    lower = upper - 1
    fraction = (target - source[lower]) / (source[upper] - source[lower])
    fraction[np.isclose(target, source[lower], rtol=0.0, atol=1e-13)] = 0.0
    return lower, upper, fraction


def bilinear_resample_finite(
    image: np.ndarray,
    source_h: np.ndarray,
    source_k: np.ndarray,
    target_h: np.ndarray,
    target_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear point sampling with finite-corner weight renormalization."""

    value = np.asarray(image, dtype=np.float64)
    if value.shape != (source_h.size, source_k.size):
        raise ValueError("image/axis shape mismatch")
    h0, h1, th = _brackets(source_h, target_h)
    k0, k1, tk = _brackets(source_k, target_k)
    output = np.empty((target_h.size, target_k.size), dtype=np.float64)
    valid = np.zeros_like(output, dtype=bool)
    for i in range(target_h.size):
        for j in range(target_k.size):
            weights = np.asarray(
                [
                    (1.0 - th[i]) * (1.0 - tk[j]),
                    th[i] * (1.0 - tk[j]),
                    (1.0 - th[i]) * tk[j],
                    th[i] * tk[j],
                ]
            )
            values = np.asarray(
                [
                    value[h0[i], k0[j]],
                    value[h1[i], k0[j]],
                    value[h0[i], k1[j]],
                    value[h1[i], k1[j]],
                ]
            )
            finite = np.isfinite(values)
            denominator = float(np.sum(weights[finite]))
            if denominator > np.finfo(float).eps:
                output[i, j] = float(np.sum(weights[finite] * values[finite]) / denominator)
                valid[i, j] = True
            else:
                output[i, j] = np.nan
    return output, valid


def nearest_resample_finite(
    image: np.ndarray,
    source_h: np.ndarray,
    source_k: np.ndarray,
    target_h: np.ndarray,
    target_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(image, dtype=np.float64)
    h_index = np.argmin(np.abs(source_h[:, None] - target_h[None, :]), axis=0)
    k_index = np.argmin(np.abs(source_k[:, None] - target_k[None, :]), axis=0)
    output = value[np.ix_(h_index, k_index)]
    return output, np.isfinite(output)


def native_fft_lattice_coefficients(
    spins: np.ndarray, reduced_h: np.ndarray, reduced_k: np.ndarray
) -> np.ndarray:
    """Positive-sign Fourier coefficients for centred 24 x 24 sites."""

    spins = np.asarray(spins)
    if spins.shape != (24, 24):
        raise ValueError("frozen lattice must be 24 x 24")
    mh = np.rint(24.0 * np.asarray(reduced_h)).astype(np.int64)
    mk = np.rint(24.0 * np.asarray(reduced_k)).astype(np.int64)
    if not np.allclose(reduced_h, mh / 24.0, rtol=0.0, atol=2e-14):
        raise ValueError("h target is not on the 1/24 native grid")
    if not np.allclose(reduced_k, mk / 24.0, rtol=0.0, atol=2e-14):
        raise ValueError("k target is not on the 1/24 native grid")
    raw = np.fft.ifft2(np.asarray(spins, dtype=np.complex128)) * spins.size
    coefficients = raw[np.mod(mh, 24)[:, None], np.mod(mk, 24)[None, :]]
    centre_phase = np.exp(
        -2j
        * np.pi
        * (
            mh[:, None] * (12.0 / 24.0)
            + mk[None, :] * (12.0 / 24.0)
        )
    )
    return coefficients * centre_phase


def separable_direct_lattice_coefficients(
    spins: np.ndarray, reduced_h: np.ndarray, reduced_k: np.ndarray
) -> np.ndarray:
    coordinates = np.arange(24, dtype=np.float64) - 12.0
    phase_h = np.exp(2j * np.pi * np.outer(reduced_h, coordinates))
    phase_k = np.exp(2j * np.pi * np.outer(reduced_k, coordinates))
    return phase_h @ np.asarray(spins, dtype=np.complex128) @ phase_k.T


def resample_map_stack(
    maps: np.ndarray,
    source_h: np.ndarray,
    source_k: np.ndarray,
    target_h: np.ndarray,
    target_k: np.ndarray,
    method: str,
) -> np.ndarray:
    output = np.empty((*maps.shape[:2], target_h.size, target_k.size), dtype=np.float64)
    function = bilinear_resample_finite if method == "bilinear" else nearest_resample_finite
    for candidate in range(maps.shape[0]):
        for replica in range(maps.shape[1]):
            value, valid = function(
                maps[candidate, replica], source_h, source_k, target_h, target_k
            )
            if not np.all(valid):
                raise RuntimeError(f"nonfinite {method} prediction resampling")
            output[candidate, replica] = value
    return output


def score_family(
    maps: np.ndarray,
    experiment: np.ndarray,
    mask: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, beta in enumerate(candidates):
        scores = []
        scales = []
        for replica in range(maps.shape[1]):
            score, scale = BASE.maximum_scaled_score(
                maps[index, replica], experiment, mask
            )
            scores.append(score)
            scales.append(scale)
        rows.append(
            {
                "schedule_index": index,
                "J_over_T": float(beta),
                "replica_chi2": scores,
                "replica_maximum_scales": scales,
                "mean_replica_chi2": float(np.mean(scores)),
                "two_sample_standard_deviations": float(2.0 * np.std(scores, ddof=1)),
            }
        )
    order = sorted(range(len(rows)), key=lambda value: rows[value]["mean_replica_chi2"])
    best = rows[order[0]]
    return {
        "best_J_over_T": best["J_over_T"],
        "best_mean_replica_chi2": best["mean_replica_chi2"],
        "order_J_over_T": [rows[index]["J_over_T"] for index in order],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--atomic-protocol", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--nxs", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--acfo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "uob100dy-scatty-native-grid-audit-protocol-v1":
        raise RuntimeError("unexpected native-grid protocol schema")
    atomic_protocol = json.loads(args.atomic_protocol.read_text(encoding="utf-8"))
    parent = protocol["frozen_parent"]
    parent_hashes = {
        "predictions": sha256(args.parent_predictions),
        "summary": sha256(args.parent_summary),
    }
    expected_parent_hashes = {
        "predictions": parent["predictions_sha256"],
        "summary": parent["summary_sha256"],
    }
    if parent_hashes != expected_parent_hashes:
        raise RuntimeError(f"frozen parent hash mismatch: {parent_hashes}")

    parent_summary = json.loads(args.parent_summary.read_text(encoding="utf-8"))
    with np.load(args.parent_predictions) as payload:
        samples = np.asarray(payload["sampled_spins"], dtype=np.int8)
        candidates = np.asarray(payload["candidates"], dtype=np.float64)
        parent_experiment = np.asarray(payload["experimental"], dtype=np.float64)
        parent_mask = np.asarray(payload["finite_mask"], dtype=bool)
        measured_h_grid = np.asarray(payload["h_grid"], dtype=np.float64)
        measured_k_grid = np.asarray(payload["k_grid"], dtype=np.float64)
        parent_maps = np.asarray(payload["acfo_replica_maps"], dtype=np.float64)
    required_shape = tuple(int(value) for value in parent["required_sample_shape"])
    if samples.shape != required_shape:
        raise RuntimeError(f"parent sample shape mismatch: {samples.shape}")
    sample_digest_before = array_sha256(samples)

    geometry = BASE.load_geometry(args.geometry)
    experiment, measured_mask, measured_h, measured_k, data_receipt = BASE.load_experimental_roi(
        args.nxs, atomic_protocol
    )
    if not np.array_equal(measured_h_grid[:, 0], measured_h):
        raise RuntimeError("parent h grid differs from the staged NeXus grid")
    if not np.array_equal(measured_k_grid[0, :], measured_k):
        raise RuntimeError("parent k grid differs from the staged NeXus grid")
    nexus_replay_matches_parent = bool(
        np.array_equal(measured_mask, parent_mask)
        and np.allclose(experiment[parent_mask], parent_experiment[parent_mask], rtol=0.0, atol=0.0)
    )

    grid = protocol["native_grid"]
    native_h = native_axis(grid["h_start"], grid["points_per_axis_including_both_boundaries"])
    native_k = native_axis(grid["k_start"], grid["points_per_axis_including_both_boundaries"])
    if not math.isclose(float(native_h[-1]), float(grid["h_stop"]), rel_tol=0.0, abs_tol=2e-14):
        raise RuntimeError("native h endpoint mismatch")
    if not math.isclose(float(native_k[-1]), float(grid["k_stop"]), rel_tol=0.0, abs_tol=2e-14):
        raise RuntimeError("native k endpoint mismatch")

    experimental_bilinear, mask_bilinear = bilinear_resample_finite(
        experiment, measured_h, measured_k, native_h, native_k
    )
    experimental_nearest, mask_nearest = nearest_resample_finite(
        experiment, measured_h, measured_k, native_h, native_k
    )

    cell = np.asarray(atomic_protocol["nexus_contract"]["expected_unit_cell_angstrom"])
    cell_nm = BASE.cell_matrix_nm(cell)
    centres_nm, centres_fractional = BASE.lattice_centres(cell_nm)
    targets = BASE.reciprocal_targets(native_h, native_k, cell_nm)
    up_factor, down_factor, per_element = BASE.motif_form_factors(
        geometry, targets, args.acfo_root
    )
    delta_factor = (up_factor - down_factor).reshape(native_h.size, native_k.size)

    native_maps = np.zeros((*samples.shape[:2], native_h.size, native_k.size), dtype=np.float64)
    for candidate in range(samples.shape[0]):
        for replica in range(samples.shape[1]):
            intensity = np.zeros((native_h.size, native_k.size), dtype=np.float64)
            for spins in samples[candidate, replica]:
                lattice = native_fft_lattice_coefficients(
                    spins, native_h - 7.0, native_k
                )
                intensity += np.abs(-delta_factor * lattice) ** 2
            native_maps[candidate, replica] = intensity / samples.shape[2]
        print(f"FFT candidate={candidate + 1}/{samples.shape[0]}", flush=True)

    subset = protocol["operators"]["acfo_subset"]
    operators = BASE.Operators(
        args.acfo_root,
        centres_nm,
        targets,
        up_factor,
        down_factor,
        padding=int(subset["harmonic_padding"]),
    )
    q_subset = np.unique(
        np.linspace(
            0,
            targets["absolute_hkl"].shape[0] - 1,
            int(protocol["operators"]["atomic_direct_subset_count"]),
            dtype=np.int64,
        )
    )
    accuracy_rows: list[dict[str, Any]] = []
    for candidate in subset["schedule_indices"]:
        for replica in subset["replica_indices"]:
            for sample_index in subset["sample_indices"]:
                spins = samples[candidate, replica, sample_index]
                fft_lattice = native_fft_lattice_coefficients(
                    spins, native_h - 7.0, native_k
                )
                direct_lattice = separable_direct_lattice_coefficients(
                    spins, native_h - 7.0, native_k
                )
                fft_amplitude = (-delta_factor * fft_lattice).ravel()
                acfo_amplitude, acfo_seconds = operators.acfo_amplitude(spins)
                atomic_direct = BASE.atomic_direct_subset(
                    centres_fractional,
                    geometry,
                    targets["absolute_hkl"],
                    per_element,
                    spins,
                    q_subset,
                )
                accuracy_rows.append(
                    {
                        "schedule_index": int(candidate),
                        "J_over_T": float(candidates[candidate]),
                        "replica_index": int(replica),
                        "sample_index": int(sample_index),
                        "q_subset_count": int(q_subset.size),
                        "fft_vs_separable_direct_relative_l2": BASE.relative_l2(
                            fft_lattice, direct_lattice
                        ),
                        "acfo_vs_fft_relative_l2": BASE.relative_l2(
                            acfo_amplitude, fft_amplitude
                        ),
                        "fft_vs_atomic_direct_relative_l2": BASE.relative_l2(
                            fft_amplitude[q_subset], atomic_direct
                        ),
                        "acfo_seconds": acfo_seconds,
                    }
                )

    native_to_measured_bilinear = resample_map_stack(
        native_maps, native_h, native_k, measured_h, measured_k, "bilinear"
    )
    native_to_measured_nearest = resample_map_stack(
        native_maps, native_h, native_k, measured_h, measured_k, "nearest"
    )
    adapters = {
        "bilinear_experiment_to_native": score_family(
            native_maps, experimental_bilinear, mask_bilinear, candidates
        ),
        "nearest_experiment_to_native": score_family(
            native_maps, experimental_nearest, mask_nearest, candidates
        ),
        "bilinear_native_prediction_to_measured": score_family(
            native_to_measured_bilinear, experiment, measured_mask, candidates
        ),
        "nearest_native_prediction_to_measured": score_family(
            native_to_measured_nearest, experiment, measured_mask, candidates
        ),
        "parent_exact_q_measured_grid": score_family(
            parent_maps, parent_experiment, parent_mask, candidates
        ),
    }

    thresholds = protocol["technical_gates"]
    max_fft_direct = max(
        row["fft_vs_separable_direct_relative_l2"] for row in accuracy_rows
    )
    max_acfo_fft = max(row["acfo_vs_fft_relative_l2"] for row in accuracy_rows)
    max_fft_atomic = max(
        row["fft_vs_atomic_direct_relative_l2"] for row in accuracy_rows
    )
    parent_best = float(parent_summary["ranking"]["acfo_best_J_over_T"])
    replay_best = float(adapters["parent_exact_q_measured_grid"]["best_J_over_T"])
    sample_digest_after = array_sha256(samples)
    gates = {
        "parent_hashes": parent_hashes == expected_parent_hashes,
        "parent_sample_shape": samples.shape == required_shape,
        "sample_bytes_unchanged": sample_digest_before == sample_digest_after,
        "native_grid_step": bool(
            np.allclose(np.diff(native_h), 1.0 / 24.0, rtol=0.0, atol=2e-14)
            and np.allclose(np.diff(native_k), 1.0 / 24.0, rtol=0.0, atol=2e-14)
        ),
        "no_lanczos": protocol["native_grid"]["lanczos_resampling"] is False,
        "fft_vs_separable_direct": max_fft_direct
        <= float(thresholds["fft_vs_separable_direct_relative_l2_max"]),
        "acfo_vs_fft": max_acfo_fft
        <= float(thresholds["acfo_vs_fft_relative_l2_max"]),
        "fft_vs_atomic_direct": max_fft_atomic
        <= float(thresholds["fft_vs_atomic_direct_relative_l2_max"]),
        "parent_exact_q_best_replayed": math.isclose(
            replay_best, parent_best, rel_tol=0.0, abs_tol=1e-15
        ),
        "nexus_replay_matches_parent": nexus_replay_matches_parent,
    }

    reference = float(protocol["ranking"]["source_reference_J_over_T"])
    half_width = float(protocol["ranking"]["source_reference_2sigma"])
    adapter_summary = {
        name: {
            "best_J_over_T": value["best_J_over_T"],
            "source_interval_recovered": bool(
                reference - half_width <= value["best_J_over_T"] <= reference + half_width
            ),
            "order_J_over_T": value["order_J_over_T"],
        }
        for name, value in adapters.items()
    }
    primary_name = protocol["measurement_adapters"]["primary"]
    summary = {
        "schema": "uob100dy-scatty-native-grid-audit-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol_sha256": sha256(args.protocol),
        "atomic_protocol_sha256": sha256(args.atomic_protocol),
        "geometry_sha256": sha256(args.geometry),
        "parent": {
            "file_hashes": parent_hashes,
            "sample_shape": list(samples.shape),
            "sample_array_sha256": sample_digest_before,
            "monte_carlo_rerun": False,
        },
        "data": data_receipt,
        "grids": {
            "measured_shape": list(experiment.shape),
            "measured_step_r_l_u": float(np.diff(measured_h)[0]),
            "native_shape": [native_h.size, native_k.size],
            "native_step_r_l_u": float(np.diff(native_h)[0]),
            "bilinear_native_finite_pixels": int(np.count_nonzero(mask_bilinear)),
            "nearest_native_finite_pixels": int(np.count_nonzero(mask_nearest)),
            "lanczos_resampling": False,
        },
        "accuracy": {
            "rows": accuracy_rows,
            "fft_vs_separable_direct_relative_l2_max": max_fft_direct,
            "acfo_vs_fft_relative_l2_max": max_acfo_fft,
            "fft_vs_atomic_direct_relative_l2_max": max_fft_atomic,
        },
        "ranking": {
            "primary_adapter": primary_name,
            "primary_best_J_over_T": adapters[primary_name]["best_J_over_T"],
            "source_reference_J_over_T": reference,
            "source_reference_2sigma": half_width,
            "adapters": adapter_summary,
        },
        "adapter_results": adapters,
        "technical_gates": gates,
        "timing": {"total_seconds": time.perf_counter() - started},
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.predictions,
        candidates=candidates,
        measured_h=measured_h,
        measured_k=measured_k,
        native_h=native_h,
        native_k=native_k,
        measured_experiment=experiment,
        measured_mask=measured_mask,
        native_experiment_bilinear=experimental_bilinear,
        native_mask_bilinear=mask_bilinear,
        native_experiment_nearest=experimental_nearest,
        native_mask_nearest=mask_nearest,
        native_fft_replica_maps=native_maps,
        native_to_measured_bilinear=native_to_measured_bilinear,
        native_to_measured_nearest=native_to_measured_nearest,
    )
    print(json.dumps({"status": summary["status"], "ranking": summary["ranking"]}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
