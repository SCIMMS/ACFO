#!/usr/bin/env python3
"""Preregistered UoB-100(Dy) decorrelation-estimator sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(
    "uob100dy_atomic_base",
    Path(__file__).with_name("run_uob100dy_atomic_decoration_validation.py"),
)
NATIVE = _load_module(
    "uob100dy_native_base",
    Path(__file__).with_name("run_uob100dy_scatty_native_grid_audit.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def autocovariances_biased(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    if centered.size < 2:
        return np.asarray([0.0])
    return np.asarray(
        [float(np.dot(centered[: centered.size - lag], centered[lag:]) / centered.size)
         for lag in range(centered.size)],
        dtype=np.float64,
    )


def geyer_initial_sequence_tau(values: np.ndarray, monotone: bool) -> float:
    """Normalized asymptotic variance using Geyer's paired sequence."""

    gamma = autocovariances_biased(values)
    if gamma.size < 2 or gamma[0] <= np.finfo(float).tiny:
        return 1.0
    pair_count = gamma.size // 2
    pairs = gamma[: 2 * pair_count].reshape(pair_count, 2).sum(axis=1)
    stop = next((index for index, value in enumerate(pairs) if value <= 0.0), pairs.size)
    positive = pairs[:stop]
    if positive.size == 0:
        return 1.0
    if monotone:
        positive = np.minimum.accumulate(positive)
    asymptotic_variance = -gamma[0] + 2.0 * float(np.sum(positive))
    return max(1.0, asymptotic_variance / gamma[0])


def estimate_decorrelation_sweeps(
    spins: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    invariants: dict[str, Any],
    estimator_name: str,
) -> dict[str, Any]:
    estimators: dict[str, Callable[[np.ndarray], float]] = {
        "geyer_ips": lambda value: geyer_initial_sequence_tau(value, monotone=False),
        "geyer_ims": lambda value: geyer_initial_sequence_tau(value, monotone=True),
    }
    if estimator_name not in estimators:
        raise ValueError(f"unsupported estimator {estimator_name}")
    tau_function = estimators[estimator_name]
    initial = int(invariants["initial_pilot_sweeps"])
    maximum = int(invariants["maximum_pilot_sweeps"])
    window_multiple = int(invariants["minimum_effective_window_multiples"])
    energies: list[float] = []
    magnetizations: list[float] = []
    accepted = 0
    target = min(initial, maximum)
    while True:
        while len(energies) < target:
            accepted += BASE.run_moves(spins, beta, spins.size, rng)
            energy, magnetization = BASE.reduced_observables(spins)
            energies.append(energy)
            magnetizations.append(magnetization)
        energy_tau = tau_function(np.asarray(energies))
        magnetization_tau = tau_function(np.asarray(magnetizations))
        tau = max(energy_tau, magnetization_tau)
        if len(energies) >= window_multiple * tau or len(energies) >= maximum:
            break
        target = min(maximum, max(target + 1, 2 * target))
    n_d_sweeps = max(1, int(math.ceil(tau)))
    return {
        "estimator": estimator_name,
        "energy_tau_sweeps": energy_tau,
        "magnetization_tau_sweeps": magnetization_tau,
        "n_d_sweeps": n_d_sweeps,
        "n_d_moves": int(n_d_sweeps * spins.size),
        "pilot_sweeps": len(energies),
        "maximum_pilot_hit": len(energies) >= maximum,
        "pilot_acceptance_fraction": accepted / float(len(energies) * spins.size),
    }


def sample_temperature(
    spins: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    invariants: dict[str, Any],
    estimator_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    decorrelation = estimate_decorrelation_sweeps(
        spins, beta, rng, invariants, estimator_name
    )
    n_d_moves = int(decorrelation["n_d_moves"])
    equilibration_moves = int(invariants["equilibration_n_d_multiples"]) * n_d_moves
    separation_moves = int(invariants["sample_separation_n_d_multiples"]) * n_d_moves
    accepted_equilibration = BASE.run_moves(spins, beta, equilibration_moves, rng)
    sample_count = int(invariants["samples_per_temperature"])
    samples = np.empty((sample_count, *spins.shape), dtype=np.int8)
    accepted_sampling = 0
    for index in range(sample_count):
        accepted_sampling += BASE.run_moves(spins, beta, separation_moves, rng)
        samples[index] = spins
    diagnostics = {
        **decorrelation,
        "beta_J_over_T": float(beta),
        "equilibration_moves": equilibration_moves,
        "moves_between_samples": separation_moves,
        "sample_count": sample_count,
        "equilibration_acceptance_fraction": accepted_equilibration
        / max(equilibration_moves, 1),
        "sampling_acceptance_fraction": accepted_sampling
        / max(sample_count * separation_moves, 1),
        "sample_energy_per_site_mean": float(
            np.mean([BASE.reduced_observables(value)[0] for value in samples])
        ),
        "sample_magnetization_per_site_mean": float(np.mean(samples)),
    }
    return samples, diagnostics


def generate_branch(
    candidates: np.ndarray,
    invariants: dict[str, Any],
    estimator_name: str,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    seeds = [int(value) for value in invariants["replica_seeds"]]
    shape = tuple(int(value) for value in invariants["lattice_shape"])
    sample_count = int(invariants["samples_per_temperature"])
    samples = np.empty(
        (candidates.size, len(seeds), sample_count, *shape), dtype=np.int8
    )
    diagnostics: list[list[dict[str, Any]]] = [[] for _ in seeds]
    for replica, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        spins = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=shape)
        for candidate, beta in enumerate(candidates):
            value, row = sample_temperature(
                spins, float(beta), rng, invariants, estimator_name
            )
            samples[candidate, replica] = value
            diagnostics[replica].append(row)
            print(
                f"MC estimator={estimator_name} replica={replica + 1}/{len(seeds)} "
                f"temperature={candidate + 1}/{candidates.size} "
                f"J/T={beta:.9f} n_d={row['n_d_sweeps']}",
                flush=True,
            )
    return samples, diagnostics


def native_maps_from_samples(
    samples: np.ndarray,
    delta_factor: np.ndarray,
    native_h: np.ndarray,
    native_k: np.ndarray,
) -> np.ndarray:
    output = np.zeros((*samples.shape[:2], native_h.size, native_k.size), dtype=np.float64)
    for candidate in range(samples.shape[0]):
        for replica in range(samples.shape[1]):
            intensity = np.zeros((native_h.size, native_k.size), dtype=np.float64)
            for spins in samples[candidate, replica]:
                lattice = NATIVE.native_fft_lattice_coefficients(
                    spins, native_h - 7.0, native_k
                )
                intensity += np.abs(-delta_factor * lattice) ** 2
            output[candidate, replica] = intensity / samples.shape[2]
    return output


def diagnostic_summary(rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    flat = [row for replica in rows for row in replica]
    return {
        "n_d_sweeps_min": min(int(row["n_d_sweeps"]) for row in flat),
        "n_d_sweeps_max": max(int(row["n_d_sweeps"]) for row in flat),
        "pilot_sweeps_min": min(int(row["pilot_sweeps"]) for row in flat),
        "pilot_sweeps_max": max(int(row["pilot_sweeps"]) for row in flat),
        "maximum_pilot_hits": sum(bool(row["maximum_pilot_hit"]) for row in flat),
    }


def parent_diagnostics(summary: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows = summary["rows"]
    replica_count = len(rows[0]["mc_by_replica"])
    return [
        [rows[candidate]["mc_by_replica"][replica] for candidate in range(len(rows))]
        for replica in range(replica_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--atomic-protocol", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--nxs", type=Path, required=True)
    parser.add_argument("--atomic-predictions", type=Path, required=True)
    parser.add_argument("--atomic-summary", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--acfo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "uob100dy-nd-estimator-sensitivity-protocol-v1":
        raise RuntimeError("unexpected estimator-sensitivity schema")
    atomic_protocol = json.loads(args.atomic_protocol.read_text(encoding="utf-8"))
    expected = protocol["frozen_parent"]
    hashes = {
        "atomic_predictions": sha256(args.atomic_predictions),
        "atomic_summary": sha256(args.atomic_summary),
        "native_predictions": sha256(args.native_predictions),
        "native_summary": sha256(args.native_summary),
    }
    expected_hashes = {
        name: expected[f"{name}_sha256"] for name in hashes
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"parent hash mismatch: {hashes}")

    atomic_summary = json.loads(args.atomic_summary.read_text(encoding="utf-8"))
    native_summary = json.loads(args.native_summary.read_text(encoding="utf-8"))
    with np.load(args.atomic_predictions) as payload:
        parent_samples = np.asarray(payload["sampled_spins"], dtype=np.int8)
        candidates = np.asarray(payload["candidates"], dtype=np.float64)
    with np.load(args.native_predictions) as payload:
        native_h = np.asarray(payload["native_h"], dtype=np.float64)
        native_k = np.asarray(payload["native_k"], dtype=np.float64)
        parent_maps = np.asarray(payload["native_fft_replica_maps"], dtype=np.float64)
        parent_experiment_bilinear = np.asarray(
            payload["native_experiment_bilinear"], dtype=np.float64
        )
        parent_mask_bilinear = np.asarray(payload["native_mask_bilinear"], dtype=bool)
        parent_experiment_nearest = np.asarray(
            payload["native_experiment_nearest"], dtype=np.float64
        )
        parent_mask_nearest = np.asarray(payload["native_mask_nearest"], dtype=bool)
    required_shape = tuple(int(value) for value in expected["required_sample_shape"])
    if parent_samples.shape != required_shape:
        raise RuntimeError(f"parent sample shape mismatch: {parent_samples.shape}")

    invariants = protocol["model_invariants"]
    model = atomic_protocol["model"]
    model_invariants_preserved = bool(
        model["lattice_shape"][:2] == invariants["lattice_shape"]
        and model["replica_seeds"] == invariants["replica_seeds"]
        and model["samples_per_temperature"] == invariants["samples_per_temperature"]
        and model["equilibration_decorrelation_multiples"]
        == invariants["equilibration_n_d_multiples"]
        and model["moves_between_samples_decorrelation_multiples"]
        == invariants["sample_separation_n_d_multiples"]
        and model["decorrelation_estimator"]["initial_pilot_sweeps"]
        == invariants["initial_pilot_sweeps"]
        and model["decorrelation_estimator"]["maximum_pilot_sweeps"]
        == invariants["maximum_pilot_sweeps"]
        and model["decorrelation_estimator"]["minimum_effective_window_multiples"]
        == invariants["minimum_effective_window_multiples"]
        and np.allclose(BASE.candidate_schedule(model), candidates, rtol=0.0, atol=0.0)
    )
    if not model_invariants_preserved:
        raise RuntimeError("frozen model invariants do not match the atomic parent")

    experiment, measured_mask, measured_h, measured_k, data_receipt = BASE.load_experimental_roi(
        args.nxs, atomic_protocol
    )
    experiment_bilinear, mask_bilinear = NATIVE.bilinear_resample_finite(
        experiment, measured_h, measured_k, native_h, native_k
    )
    experiment_nearest, mask_nearest = NATIVE.nearest_resample_finite(
        experiment, measured_h, measured_k, native_h, native_k
    )
    experiment_replay = bool(
        np.array_equal(mask_bilinear, parent_mask_bilinear)
        and np.array_equal(mask_nearest, parent_mask_nearest)
        and np.allclose(
            experiment_bilinear[mask_bilinear],
            parent_experiment_bilinear[parent_mask_bilinear],
            rtol=0.0,
            atol=0.0,
        )
        and np.allclose(
            experiment_nearest[mask_nearest],
            parent_experiment_nearest[parent_mask_nearest],
            rtol=0.0,
            atol=0.0,
        )
    )

    geometry = BASE.load_geometry(args.geometry)
    cell = np.asarray(atomic_protocol["nexus_contract"]["expected_unit_cell_angstrom"])
    cell_nm = BASE.cell_matrix_nm(cell)
    targets = BASE.reciprocal_targets(native_h, native_k, cell_nm)
    up_factor, down_factor, _ = BASE.motif_form_factors(geometry, targets, args.acfo_root)
    delta_factor = (up_factor - down_factor).reshape(native_h.size, native_k.size)

    branch_samples: dict[str, np.ndarray] = {}
    branch_diagnostics: dict[str, list[list[dict[str, Any]]]] = {}
    branch_maps: dict[str, np.ndarray] = {"parent_positive_lag": parent_maps}
    for estimator in ("geyer_ips", "geyer_ims"):
        samples, diagnostics = generate_branch(candidates, invariants, estimator)
        branch_samples[estimator] = samples
        branch_diagnostics[estimator] = diagnostics
        branch_maps[estimator] = native_maps_from_samples(
            samples, delta_factor, native_h, native_k
        )
        print(f"MAP estimator={estimator} complete", flush=True)

    branch_diagnostics["parent_positive_lag"] = parent_diagnostics(atomic_summary)
    results: dict[str, Any] = {}
    reference = float(protocol["scattering_and_ranking"]["source_reference_J_over_T"])
    half_width = float(protocol["scattering_and_ranking"]["source_reference_2sigma"])
    for estimator, maps in branch_maps.items():
        primary = NATIVE.score_family(
            maps, experiment_bilinear, mask_bilinear, candidates
        )
        nearest = NATIVE.score_family(
            maps, experiment_nearest, mask_nearest, candidates
        )
        results[estimator] = {
            "primary_bilinear": primary,
            "sensitivity_nearest": nearest,
            "primary_source_interval_recovered": bool(
                reference - half_width <= primary["best_J_over_T"] <= reference + half_width
            ),
            "nearest_source_interval_recovered": bool(
                reference - half_width <= nearest["best_J_over_T"] <= reference + half_width
            ),
            "decorrelation": diagnostic_summary(branch_diagnostics[estimator]),
        }

    parent_replay_best = float(results["parent_positive_lag"]["primary_bilinear"]["best_J_over_T"])
    expected_parent_best = float(native_summary["ranking"]["primary_best_J_over_T"])
    new_maximum_hits = sum(
        results[name]["decorrelation"]["maximum_pilot_hits"]
        for name in ("geyer_ips", "geyer_ims")
    )
    all_maps_finite = all(np.all(np.isfinite(value)) for value in branch_maps.values())
    gates = {
        "all_parent_hashes": hashes == expected_hashes,
        "model_invariants_preserved": model_invariants_preserved,
        "new_sample_shape": all(
            value.shape == required_shape for value in branch_samples.values()
        ),
        "new_maximum_pilot_hits": new_maximum_hits
        <= int(protocol["technical_gates"]["new_maximum_pilot_hits"]),
        "all_maps_finite": all_maps_finite,
        "parent_native_best_replayed": math.isclose(
            parent_replay_best, expected_parent_best, rel_tol=0.0, abs_tol=1e-15
        ),
        "no_lanczos": protocol["scattering_and_ranking"]["lanczos_resampling"] is False,
        "experiment_adapter_replayed": experiment_replay,
    }
    compact = {
        estimator: {
            "primary_best_J_over_T": value["primary_bilinear"]["best_J_over_T"],
            "nearest_best_J_over_T": value["sensitivity_nearest"]["best_J_over_T"],
            "primary_source_interval_recovered": value[
                "primary_source_interval_recovered"
            ],
            "nearest_source_interval_recovered": value[
                "nearest_source_interval_recovered"
            ],
            "decorrelation": value["decorrelation"],
        }
        for estimator, value in results.items()
    }
    summary = {
        "schema": "uob100dy-nd-estimator-sensitivity-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol_sha256": sha256(args.protocol),
        "atomic_protocol_sha256": sha256(args.atomic_protocol),
        "geometry_sha256": sha256(args.geometry),
        "parent_hashes": hashes,
        "data": data_receipt,
        "estimators": compact,
        "full_results": results,
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
        native_h=native_h,
        native_k=native_k,
        native_experiment_bilinear=experiment_bilinear,
        native_mask_bilinear=mask_bilinear,
        native_experiment_nearest=experiment_nearest,
        native_mask_nearest=mask_nearest,
        geyer_ips_samples=branch_samples["geyer_ips"],
        geyer_ims_samples=branch_samples["geyer_ims"],
        parent_positive_lag_maps=branch_maps["parent_positive_lag"],
        geyer_ips_maps=branch_maps["geyer_ips"],
        geyer_ims_maps=branch_maps["geyer_ims"],
    )
    print(json.dumps({"status": summary["status"], "estimators": compact}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
