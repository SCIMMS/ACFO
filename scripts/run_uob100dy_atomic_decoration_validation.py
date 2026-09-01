#!/usr/bin/env python3
"""Source-faithful UoB-100(Dy) atomic-decoration ranking validation.

The published geometry and sampling counts are reproduced. The exact custom
decorrelation estimator was not published, so this runner records a transparent
independent estimator and preserves the protocol's conservative claim boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(reference)) / denominator)


def canonical_geometry_hash(atoms: list[list[Any]]) -> str:
    normalized = sorted(
        [
            str(element),
            *[0.0 if float(value) == 0.0 else float(value) for value in (x, y, z)],
        ]
        for element, x, y, z in atoms
    )
    canonical = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def load_geometry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "uob100dy-spin-cluster-geometries-v1":
        raise RuntimeError("unexpected geometry schema")
    up_atoms = payload["spin_up"]["atoms"]
    down_atoms = [
        [row[0], *[0.0 if float(value) == 0.0 else -float(value) for value in row[1:]]]
        for row in up_atoms
    ]
    if len(up_atoms) != int(payload["expected_atom_count_per_state"]):
        raise RuntimeError("spin-up atom count mismatch")
    composition = Counter(str(row[0]) for row in up_atoms)
    if composition != Counter(payload["expected_composition_per_state"]):
        raise RuntimeError(f"spin-up composition mismatch: {dict(composition)}")
    hashes = {
        "spin_up": canonical_geometry_hash(up_atoms),
        "spin_down": canonical_geometry_hash(down_atoms),
    }
    expected = {
        "spin_up": payload["spin_up"]["canonical_multiset_sha256"],
        "spin_down": payload["spin_down"]["canonical_multiset_sha256"],
    }
    if hashes != expected:
        raise RuntimeError(f"geometry hash mismatch: {hashes}")
    return {
        "payload": payload,
        "elements": np.asarray([str(row[0]) for row in up_atoms]),
        "up_fractional": np.asarray([row[1:] for row in up_atoms], dtype=np.float64),
        "down_fractional": np.asarray([row[1:] for row in down_atoms], dtype=np.float64),
        "hashes": hashes,
        "composition": {str(key): int(value) for key, value in sorted(composition.items())},
    }


def exact_axis_index(axis: np.ndarray, value: float, tolerance: float = 1e-9) -> int:
    index = int(np.argmin(np.abs(axis - value)))
    if abs(float(axis[index]) - value) > tolerance:
        raise RuntimeError(f"axis does not contain required coordinate {value}")
    return index


def validate_axis(axis: np.ndarray, contract: dict[str, Any], name: str) -> None:
    count = int(contract["expected_shape"][0])
    expected = np.linspace(
        float(contract["expected_axis_start"]),
        float(contract["expected_axis_stop"]),
        count,
    )
    if axis.shape != (count,) or not np.allclose(axis, expected, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{name} axis violates the frozen contract")


def load_experimental_roi(
    nxs_path: Path, protocol: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    nexus = protocol["nexus_contract"]
    prep = protocol["experimental_preprocessing"]
    with h5py.File(nxs_path, "r") as handle:
        axes = {
            name: np.asarray(handle[path], dtype=np.float64)
            for name, path in nexus["axis_paths"].items()
        }
        for name, axis in axes.items():
            validate_axis(axis, nexus, name)
        signal = handle[nexus["signal_path"]]
        if list(signal.shape) != list(nexus["expected_shape"]):
            raise RuntimeError(f"signal shape mismatch: {signal.shape}")
        cell = np.asarray(handle[nexus["unit_cell_path"]], dtype=np.float64)
        expected_cell = np.asarray(nexus["expected_unit_cell_angstrom"], dtype=np.float64)
        if cell.shape != (6,) or not np.allclose(cell, expected_cell, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"unit-cell mismatch: {cell.tolist()}")
        center = np.asarray(prep["roi_center_hkl"], dtype=np.float64)
        half = np.asarray(prep["roi_half_width_hkl"], dtype=np.float64)
        h0 = exact_axis_index(axes["h"], center[0] - half[0])
        h1 = exact_axis_index(axes["h"], center[0] + half[0])
        k0 = exact_axis_index(axes["k"], center[1] - half[1])
        k1 = exact_axis_index(axes["k"], center[1] + half[1])
        offsets = [float(prep["background_l_offsets"][0]), 0.0, float(prep["background_l_offsets"][1])]
        l_indices = [exact_axis_index(axes["l"], center[2] + offset) for offset in offsets]
        slabs = [
            np.asarray(signal[index, k0 : k1 + 1, h0 : h1 + 1], dtype=np.float64).T
            for index in l_indices
        ]
    experimental = slabs[1] - 0.5 * (slabs[0] + slabs[2])
    finite = np.isfinite(experimental)
    if np.count_nonzero(finite) < 0.8 * experimental.size:
        raise RuntimeError("fewer than 80% of frozen ROI pixels are finite")
    receipt = {
        "nxs_path": str(nxs_path),
        "nxs_bytes": nxs_path.stat().st_size,
        "nxs_sha256": sha256(nxs_path),
        "loaded_slab_shape": list(slabs[0].shape),
        "loaded_voxels": int(3 * slabs[0].size),
        "finite_pixels": int(np.count_nonzero(finite)),
        "total_pixels": int(experimental.size),
        "experimental_min": float(np.nanmin(experimental)),
        "experimental_max": float(np.nanmax(experimental)),
        "unit_cell_angstrom": cell.tolist(),
    }
    return experimental, finite, axes["h"][h0 : h1 + 1], axes["k"][k0 : k1 + 1], receipt


def cell_matrix_nm(cell_angstrom: np.ndarray) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = np.asarray(cell_angstrom, dtype=float)
    alpha, beta, gamma = np.deg2rad((alpha_deg, beta_deg, gamma_deg))
    ca, cb, cg = np.cos((alpha, beta, gamma))
    sg = math.sin(gamma)
    volume_factor = math.sqrt(1.0 + 2.0 * ca * cb * cg - ca**2 - cb**2 - cg**2)
    return 0.1 * np.asarray(
        [[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [c * cb, c * (ca - cb * cg) / sg, c * volume_factor / sg]],
        dtype=np.float64,
    )


def lattice_centres(cell_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.indices((24, 24, 2), dtype=np.int64).reshape(3, -1).T
    fractional = indices - np.asarray((12, 12, 1))[None, :]
    return np.ascontiguousarray(fractional @ cell_nm), np.ascontiguousarray(fractional, dtype=np.float64)


def reciprocal_targets(h_values: np.ndarray, k_values: np.ndarray, cell_nm: np.ndarray) -> dict[str, np.ndarray]:
    h_grid, k_grid = np.meshgrid(h_values, k_values, indexing="ij")
    absolute = np.column_stack((h_grid.ravel(), k_grid.ravel(), np.full(h_grid.size, 0.5)))
    reduced = absolute - np.asarray((7.0, 0.0, 0.0))[None, :]
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell_nm).T
    return {
        "h_grid": h_grid,
        "k_grid": k_grid,
        "absolute_hkl": absolute,
        "reduced_hkl": reduced,
        "q_absolute_nm": np.ascontiguousarray(absolute @ reciprocal),
        "q_reduced_nm": np.ascontiguousarray(reduced @ reciprocal),
    }


def candidate_schedule(model: dict[str, Any]) -> np.ndarray:
    schedule = model["annealing_schedule"]
    return np.geomspace(
        1.0 / float(schedule["temperature_over_J_start"]),
        1.0 / float(schedule["temperature_over_J_stop"]),
        int(schedule["points"]),
    )


try:
    from numba import njit

    MC_BACKEND = "numba"
except ImportError:  # pragma: no cover - production package includes numba
    MC_BACKEND = "python-fallback"

    def njit(*_args: Any, **_kwargs: Any):
        def decorate(function):
            return function

        return decorate


@njit(cache=False)
def _metropolis_precomputed(
    spins: np.ndarray,
    beta: float,
    rows: np.ndarray,
    columns: np.ndarray,
    uniforms: np.ndarray,
) -> int:
    accepted = 0
    n0, n1 = spins.shape
    for move in range(rows.size):
        i = rows[move]
        j = columns[move]
        neighbour_sum = (
            spins[(i + 1) % n0, j]
            + spins[(i - 1) % n0, j]
            + spins[i, (j + 1) % n1]
            + spins[i, (j - 1) % n1]
            + spins[(i + 1) % n0, (j - 1) % n1]
            + spins[(i - 1) % n0, (j + 1) % n1]
        )
        delta = 2.0 * beta * spins[i, j] * neighbour_sum
        if delta <= 0.0 or uniforms[move] < math.exp(-delta):
            spins[i, j] = -spins[i, j]
            accepted += 1
    return accepted


def run_moves(
    spins: np.ndarray,
    beta: float,
    move_count: int,
    rng: np.random.Generator,
    chunk_size: int = 65536,
) -> int:
    accepted = 0
    remaining = int(move_count)
    while remaining:
        count = min(remaining, chunk_size)
        rows = rng.integers(0, spins.shape[0], size=count, dtype=np.int64)
        columns = rng.integers(0, spins.shape[1], size=count, dtype=np.int64)
        uniforms = rng.random(count)
        accepted += int(_metropolis_precomputed(spins, beta, rows, columns, uniforms))
        remaining -= count
    return accepted


def reduced_observables(spins: np.ndarray) -> tuple[float, float]:
    pair_sum = (
        np.sum(spins * np.roll(spins, -1, axis=0))
        + np.sum(spins * np.roll(spins, -1, axis=1))
        + np.sum(spins * np.roll(np.roll(spins, -1, axis=0), 1, axis=1))
    )
    return -float(pair_sum) / spins.size, float(np.mean(spins))


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    if centered.size < 2:
        return 1.0
    variance = float(np.dot(centered, centered) / centered.size)
    if variance <= np.finfo(float).tiny:
        return 1.0
    correlation_sum = 0.0
    for lag in range(1, centered.size):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / ((centered.size - lag) * variance))
        if not np.isfinite(rho) or rho <= 0.0:
            break
        correlation_sum += rho
    return max(1.0, 1.0 + 2.0 * correlation_sum)


def estimate_decorrelation_sweeps(
    spins: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    estimator: dict[str, Any],
) -> dict[str, Any]:
    initial = int(estimator["initial_pilot_sweeps"])
    maximum = int(estimator["maximum_pilot_sweeps"])
    window_multiple = int(estimator["minimum_effective_window_multiples"])
    energies: list[float] = []
    magnetizations: list[float] = []
    accepted = 0
    target = min(initial, maximum)
    while True:
        while len(energies) < target:
            accepted += run_moves(spins, beta, spins.size, rng)
            energy, magnetization = reduced_observables(spins)
            energies.append(energy)
            magnetizations.append(magnetization)
        energy_tau = integrated_autocorrelation_time(np.asarray(energies))
        magnetization_tau = integrated_autocorrelation_time(np.asarray(magnetizations))
        tau = max(energy_tau, magnetization_tau)
        if len(energies) >= window_multiple * tau or len(energies) >= maximum:
            break
        target = min(maximum, max(target + 1, 2 * target))
    n_d_sweeps = max(1, int(math.ceil(tau)))
    return {
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
    model: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    decorrelation = estimate_decorrelation_sweeps(
        spins, beta, rng, model["decorrelation_estimator"]
    )
    n_d_moves = int(decorrelation["n_d_moves"])
    equilibration_moves = int(model["equilibration_decorrelation_multiples"]) * n_d_moves
    separation_moves = int(model["moves_between_samples_decorrelation_multiples"]) * n_d_moves
    accepted_equilibration = run_moves(spins, beta, equilibration_moves, rng)
    sample_count = int(model["samples_per_temperature"])
    samples = np.empty((sample_count, *spins.shape), dtype=np.int8)
    accepted_sampling = 0
    for index in range(sample_count):
        accepted_sampling += run_moves(spins, beta, separation_moves, rng)
        samples[index] = spins
    diagnostics = {
        **decorrelation,
        "beta_J_over_T": float(beta),
        "equilibration_moves": equilibration_moves,
        "moves_between_samples": separation_moves,
        "sample_count": sample_count,
        "equilibration_acceptance_fraction": accepted_equilibration / max(equilibration_moves, 1),
        "sampling_acceptance_fraction": accepted_sampling / max(sample_count * separation_moves, 1),
        "sample_energy_per_site_mean": float(np.mean([reduced_observables(value)[0] for value in samples])),
        "sample_magnetization_per_site_mean": float(np.mean(samples)),
    }
    return samples, diagnostics


def generate_samples(model: dict[str, Any]) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    candidates = candidate_schedule(model)
    seeds = [int(value) for value in model["replica_seeds"]]
    sample_count = int(model["samples_per_temperature"])
    samples = np.empty((len(candidates), len(seeds), sample_count, 24, 24), dtype=np.int8)
    diagnostics: list[list[dict[str, Any]]] = [[] for _ in seeds]
    for replica_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        spins = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(24, 24))
        for candidate_index, beta in enumerate(candidates):
            temperature_samples, row = sample_temperature(spins, float(beta), rng, model)
            samples[candidate_index, replica_index] = temperature_samples
            diagnostics[replica_index].append(row)
            print(
                f"MC replica={replica_index + 1}/{len(seeds)} "
                f"temperature={candidate_index + 1}/{len(candidates)} "
                f"J/T={beta:.9f} n_d={row['n_d_sweeps']} sweeps",
                flush=True,
            )
    return samples, diagnostics


def motif_form_factors(
    geometry: dict[str, Any],
    targets: dict[str, np.ndarray],
    acfo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source_path = acfo_root / "src"
    python_path = source_path if (source_path / "waxs_cake").is_dir() else acfo_root
    if not (python_path / "waxs_cake").is_dir():
        raise RuntimeError(f"ACFO Python package not found under {acfo_root}")
    if str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))
    from waxs_cake.xray_form_factors import xray_f0_form_factors

    q_norm = np.linalg.norm(targets["q_absolute_nm"], axis=1)
    per_element = xray_f0_form_factors(geometry["elements"], q_norm)
    atom_factors = np.stack([per_element[str(element)] for element in geometry["elements"]])

    def factor(coordinates: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * 2.0 * np.pi * (coordinates @ targets["absolute_hkl"].T))
        return np.sum(atom_factors * phase, axis=0)

    return factor(geometry["up_fractional"]), factor(geometry["down_fractional"]), per_element


def expanded_state(spins: np.ndarray) -> np.ndarray:
    return np.stack((spins, -spins), axis=2).astype(np.float64)


class Operators:
    def __init__(
        self,
        acfo_root: Path,
        centres_nm: np.ndarray,
        targets: dict[str, np.ndarray],
        up_factor: np.ndarray,
        down_factor: np.ndarray,
        padding: int,
    ) -> None:
        source_path = acfo_root / "src"
        python_path = source_path if (source_path / "waxs_cake").is_dir() else acfo_root
        if str(python_path) not in sys.path:
            sys.path.insert(0, str(python_path))
        from waxs_cake.exact_harmonic import PreparedExactCoordinateHarmonicPlan

        import finufft

        self.target_count = targets["q_reduced_nm"].shape[0]
        source_coordinates = np.ascontiguousarray(np.vstack((centres_nm, centres_nm)))
        q_reduced = targets["q_reduced_nm"]
        q_perp = np.hypot(q_reduced[:, 0], q_reduced[:, 1])
        q_z = q_reduced[:, 2]
        phi = np.arctan2(q_reduced[:, 1], q_reduced[:, 0])
        setup_start = time.perf_counter()
        self.acfo = PreparedExactCoordinateHarmonicPlan(
            source_coordinates,
            q_perp,
            q_z,
            phi,
            element_indices=np.concatenate(
                (
                    np.zeros(centres_nm.shape[0], dtype=np.int64),
                    np.ones(centres_nm.shape[0], dtype=np.int64),
                )
            ),
            form_factors=np.ascontiguousarray(np.vstack((up_factor, down_factor))),
            harmonic_margin=padding,
            prepare_direct_basis=True,
            coefficient_backend="cached_phase",
        )
        self.acfo_setup_seconds = time.perf_counter() - setup_start
        self.delta_factor = np.ascontiguousarray(up_factor - down_factor)
        setup_start = time.perf_counter()
        self.finufft = finufft.Plan(
            2,
            (24, 24),
            n_trans=1,
            eps=1e-12,
            isign=+1,
            modeord=0,
            nthreads=1,
        )
        self.finufft.setpts(
            np.ascontiguousarray(2.0 * np.pi * targets["reduced_hkl"][:, 0]),
            np.ascontiguousarray(2.0 * np.pi * targets["reduced_hkl"][:, 1]),
        )
        self.finufft_setup_seconds = time.perf_counter() - setup_start

    @staticmethod
    def source_weights(spins: np.ndarray) -> np.ndarray:
        expanded = expanded_state(spins).reshape(-1)
        return np.ascontiguousarray(
            np.concatenate((0.5 * expanded, -0.5 * expanded)), dtype=np.complex128
        )

    def acfo_amplitude(self, spins: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        full, _ = self.acfo.execute(
            atom_weights=self.source_weights(spins),
            synthesis_backend="direct",
        )
        elapsed = time.perf_counter() - start
        indices = np.arange(self.target_count)
        return np.asarray(full)[indices, indices], elapsed

    def finufft_amplitude(self, spins: np.ndarray) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        lattice = np.asarray(
            self.finufft.execute(np.ascontiguousarray(spins, dtype=np.complex128)),
            dtype=np.complex128,
        )
        # The centred layer indices are z=(-1, 0). At l=0.5, their phases are
        # (-1, +1), while the second layer has the opposite spin.
        return -self.delta_factor * lattice, time.perf_counter() - start


def atomic_direct_subset(
    centres_fractional: np.ndarray,
    geometry: dict[str, Any],
    absolute_hkl: np.ndarray,
    per_element_factors: dict[str, np.ndarray],
    spins: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    expanded = expanded_state(spins).reshape(-1)
    weights = (0.5 * expanded, 0.5 * expanded)
    motifs = (geometry["up_fractional"], geometry["down_fractional"])
    signs = (1.0, -1.0)
    output = np.zeros(indices.size, dtype=np.complex128)
    for output_index, q_index in enumerate(indices):
        hkl = absolute_hkl[q_index]
        atom_factors = np.asarray(
            [per_element_factors[str(element)][q_index] for element in geometry["elements"]],
            dtype=np.float64,
        )
        value = 0.0j
        for motif, centre_weights, sign in zip(motifs, weights, signs, strict=True):
            phase_centres = np.exp(1j * 2.0 * np.pi * (centres_fractional @ hkl))
            phase_motif = np.exp(1j * 2.0 * np.pi * (motif @ hkl))
            value += sign * np.sum(centre_weights * phase_centres) * np.sum(atom_factors * phase_motif)
        output[output_index] = value
    return output


def maximum_scaled_score(
    prediction: np.ndarray, experiment: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    x = np.asarray(prediction, dtype=np.float64)[mask]
    y = np.asarray(experiment, dtype=np.float64)[mask]
    prediction_max = float(np.max(x))
    experimental_max = float(np.max(y))
    if prediction_max <= 0.0 or experimental_max <= 0.0:
        raise RuntimeError("maximum normalization requires positive maxima")
    scale = experimental_max / prediction_max
    return float(np.sum((scale * x - y) ** 2)), scale


def heldout_score(
    prediction: np.ndarray, experiment: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    parity = np.indices(prediction.shape).sum(axis=0) % 2
    train = mask & (parity == 0)
    test = mask & (parity == 1)
    x_train = prediction[train]
    y_train = experiment[train]
    denominator = float(np.dot(x_train, x_train))
    scale = max(0.0, float(np.dot(x_train, y_train)) / max(denominator, np.finfo(float).tiny))
    x_test = scale * prediction[test]
    y_test = experiment[test]
    residual = x_test - y_test
    nrmse = float(np.sqrt(np.mean(residual**2)) / max(float(np.std(y_test)), np.finfo(float).tiny))
    correlation = (
        float("nan")
        if float(np.std(x_test)) == 0.0 or float(np.std(y_test)) == 0.0
        else float(np.corrcoef(x_test, y_test)[0, 1])
    )
    return {
        "train_scale": scale,
        "test_normalized_rmse": nrmse,
        "test_pearson": correlation,
        "train_pixels": int(np.count_nonzero(train)),
        "test_pixels": int(np.count_nonzero(test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--nxs", type=Path, required=True)
    parser.add_argument("--acfo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "uob100dy-atomic-decoration-validation-protocol-v1":
        raise RuntimeError("unexpected protocol schema")
    geometry = load_geometry(args.geometry)
    experiment, finite, h_values, k_values, data_receipt = load_experimental_roi(
        args.nxs, protocol
    )
    cell = np.asarray(protocol["nexus_contract"]["expected_unit_cell_angstrom"], dtype=float)
    cell_nm = cell_matrix_nm(cell)
    centres_nm, centres_fractional = lattice_centres(cell_nm)
    targets = reciprocal_targets(h_values, k_values, cell_nm)
    up_factor, down_factor, per_element_factors = motif_form_factors(
        geometry, targets, args.acfo_root
    )
    padding = int(protocol["operators"]["acfo"]["harmonic_padding"])
    operators = Operators(
        args.acfo_root,
        centres_nm,
        targets,
        up_factor,
        down_factor,
        padding,
    )

    total_start = time.perf_counter()
    samples, mc_diagnostics = generate_samples(protocol["model"])
    candidates = candidate_schedule(protocol["model"])
    candidate_count, replica_count, sample_count = samples.shape[:3]
    pixel_count = experiment.size
    acfo_replica_maps = np.zeros((candidate_count, replica_count, pixel_count), dtype=np.float64)
    finufft_replica_maps = np.zeros_like(acfo_replica_maps)
    acfo_times = np.zeros((candidate_count, replica_count), dtype=np.float64)
    finufft_times = np.zeros_like(acfo_times)
    direct_indices = np.unique(
        np.linspace(
            0,
            pixel_count - 1,
            int(protocol["operators"]["direct_reference"]["q_subset_count"]),
            dtype=np.int64,
        )
    )
    direct_schedule_indices = {
        int(value) for value in protocol["operators"]["direct_reference"]["schedule_indices"]
    }
    accuracy_rows: list[dict[str, Any]] = []

    for candidate_index, beta in enumerate(candidates):
        for replica_index in range(replica_count):
            acfo_intensity = np.zeros(pixel_count, dtype=np.float64)
            finufft_intensity = np.zeros(pixel_count, dtype=np.float64)
            for sample_index in range(sample_count):
                spins = samples[candidate_index, replica_index, sample_index]
                acfo_amplitude, acfo_seconds = operators.acfo_amplitude(spins)
                finufft_amplitude, finufft_seconds = operators.finufft_amplitude(spins)
                acfo_intensity += np.abs(acfo_amplitude) ** 2
                finufft_intensity += np.abs(finufft_amplitude) ** 2
                acfo_times[candidate_index, replica_index] += acfo_seconds
                finufft_times[candidate_index, replica_index] += finufft_seconds
                if (
                    candidate_index in direct_schedule_indices
                    and replica_index == 0
                    and sample_index == 0
                ):
                    direct = atomic_direct_subset(
                        centres_fractional,
                        geometry,
                        targets["absolute_hkl"],
                        per_element_factors,
                        spins,
                        direct_indices,
                    )
                    accuracy_rows.append(
                        {
                            "schedule_index": candidate_index,
                            "J_over_T": float(beta),
                            "replica_index": replica_index,
                            "sample_index": sample_index,
                            "q_subset_count": int(direct_indices.size),
                            "acfo_vs_atomic_direct_relative_l2": relative_l2(
                                acfo_amplitude[direct_indices], direct
                            ),
                            "finufft_vs_atomic_direct_relative_l2": relative_l2(
                                finufft_amplitude[direct_indices], direct
                            ),
                        }
                    )
                if (sample_index + 1) % 20 == 0:
                    print(
                        f"OP candidate={candidate_index + 1}/{candidate_count} "
                        f"replica={replica_index + 1}/{replica_count} "
                        f"sample={sample_index + 1}/{sample_count}",
                        flush=True,
                    )
            acfo_replica_maps[candidate_index, replica_index] = acfo_intensity / sample_count
            finufft_replica_maps[candidate_index, replica_index] = finufft_intensity / sample_count

    prediction_errors = [
        relative_l2(acfo_replica_maps[i, j], finufft_replica_maps[i, j])
        for i in range(candidate_count)
        for j in range(replica_count)
    ]
    rows: list[dict[str, Any]] = []
    for candidate_index, beta in enumerate(candidates):
        acfo_scores = []
        finufft_scores = []
        acfo_scales = []
        finufft_scales = []
        for replica_index in range(replica_count):
            acfo_score, acfo_scale = maximum_scaled_score(
                acfo_replica_maps[candidate_index, replica_index].reshape(experiment.shape),
                experiment,
                finite,
            )
            finufft_score, finufft_scale = maximum_scaled_score(
                finufft_replica_maps[candidate_index, replica_index].reshape(experiment.shape),
                experiment,
                finite,
            )
            acfo_scores.append(acfo_score)
            finufft_scores.append(finufft_score)
            acfo_scales.append(acfo_scale)
            finufft_scales.append(finufft_scale)
        acfo_mean_map = np.mean(acfo_replica_maps[candidate_index], axis=0).reshape(experiment.shape)
        finufft_mean_map = np.mean(finufft_replica_maps[candidate_index], axis=0).reshape(experiment.shape)
        rows.append(
            {
                "schedule_index": candidate_index,
                "J_over_T": float(beta),
                "temperature_over_J": float(1.0 / beta),
                "acfo": {
                    "replica_chi2": acfo_scores,
                    "replica_maximum_scales": acfo_scales,
                    "mean_replica_chi2": float(np.mean(acfo_scores)),
                    "two_sample_standard_deviations": float(
                        2.0 * np.std(acfo_scores, ddof=1)
                    ),
                    "heldout_mean_map": heldout_score(acfo_mean_map, experiment, finite),
                    "apply_seconds_by_replica": acfo_times[candidate_index].tolist(),
                },
                "finufft": {
                    "replica_chi2": finufft_scores,
                    "replica_maximum_scales": finufft_scales,
                    "mean_replica_chi2": float(np.mean(finufft_scores)),
                    "two_sample_standard_deviations": float(
                        2.0 * np.std(finufft_scores, ddof=1)
                    ),
                    "heldout_mean_map": heldout_score(finufft_mean_map, experiment, finite),
                    "apply_seconds_by_replica": finufft_times[candidate_index].tolist(),
                },
                "mc_by_replica": [
                    mc_diagnostics[replica_index][candidate_index]
                    for replica_index in range(replica_count)
                ],
            }
        )

    acfo_order = sorted(
        range(candidate_count), key=lambda index: (rows[index]["acfo"]["mean_replica_chi2"], index)
    )
    finufft_order = sorted(
        range(candidate_count), key=lambda index: (rows[index]["finufft"]["mean_replica_chi2"], index)
    )
    gates_contract = protocol["technical_gates"]
    max_acfo_direct = max(
        row["acfo_vs_atomic_direct_relative_l2"] for row in accuracy_rows
    )
    max_finufft_direct = max(
        row["finufft_vs_atomic_direct_relative_l2"] for row in accuracy_rows
    )
    gates = {
        "geometry_canonical_hashes": True,
        "sample_count_per_temperature": sample_count
        == int(gates_contract["sample_count_per_temperature"]),
        "replica_count": replica_count == int(gates_contract["replica_count"]),
        "acfo_vs_atomic_direct": max_acfo_direct
        <= float(gates_contract["acfo_vs_atomic_direct_relative_l2_max"]),
        "finufft_vs_atomic_direct": max_finufft_direct
        <= float(gates_contract["finufft_vs_atomic_direct_relative_l2_max"]),
        "acfo_vs_finufft_prediction": max(prediction_errors)
        <= float(gates_contract["acfo_vs_finufft_prediction_relative_l2_max"]),
        "same_top_candidate": acfo_order[0] == finufft_order[0],
        "no_interpolation": protocol["experimental_preprocessing"]["interpolation"] == "none",
        "no_lanczos": True,
        "exact_qR_plus_48": padding == 48,
        "miller_margin_32": int(protocol["operators"]["acfo"]["miller_recurrence_margin"]) == 32,
    }
    acfo_best = float(candidates[acfo_order[0]])
    reference = float(protocol["ranking"]["source_reference_J_over_T"])
    reference_half_width = float(protocol["ranking"]["source_reference_2sigma"])
    summary = {
        "schema": "uob100dy-atomic-decoration-validation-summary-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol_sha256": sha256(args.protocol),
        "geometry_sha256": sha256(args.geometry),
        "data": data_receipt,
        "geometry": {
            "source_tables": geometry["payload"]["source"]["actual_tables"],
            "canonical_hashes": geometry["hashes"],
            "composition_per_state": geometry["composition"],
            "atoms_per_state": int(geometry["elements"].size),
            "node_count": int(centres_nm.shape[0]),
            "prepared_source_count": int(2 * centres_nm.shape[0]),
            "target_count": pixel_count,
            "roi_shape": list(experiment.shape),
            "cell_matrix_nm": cell_nm.tolist(),
            "harmonic_padding": padding,
            "miller_margin": 32,
            "cutoff_min": int(np.min(operators.acfo.cutoffs)),
            "cutoff_max": int(np.max(operators.acfo.cutoffs)),
        },
        "monte_carlo": {
            "backend": MC_BACKEND,
            "schedule_J_over_T": candidates.tolist(),
            "replica_seeds": protocol["model"]["replica_seeds"],
            "samples_per_temperature": sample_count,
            "maximum_pilot_hit_count": int(
                sum(
                    row["maximum_pilot_hit"]
                    for replica_rows in mc_diagnostics
                    for row in replica_rows
                )
            ),
            "estimator_status": protocol["model"]["decorrelation_estimator"]["status"],
        },
        "accuracy": {
            "direct_subset_indices": direct_indices.tolist(),
            "direct_checks": accuracy_rows,
            "max_acfo_vs_atomic_direct_relative_l2": max_acfo_direct,
            "max_finufft_vs_atomic_direct_relative_l2": max_finufft_direct,
            "max_acfo_vs_finufft_prediction_relative_l2": max(prediction_errors),
        },
        "ranking": {
            "acfo_order": [float(candidates[index]) for index in acfo_order],
            "finufft_order": [float(candidates[index]) for index in finufft_order],
            "acfo_best_J_over_T": acfo_best,
            "finufft_best_J_over_T": float(candidates[finufft_order[0]]),
            "source_reference_J_over_T": reference,
            "source_reference_2sigma": reference_half_width,
            "source_interval_recovered": reference - reference_half_width
            <= acfo_best
            <= reference + reference_half_width,
        },
        "timing": {
            "acfo_setup_seconds": operators.acfo_setup_seconds,
            "finufft_setup_seconds": operators.finufft_setup_seconds,
            "acfo_apply_seconds": float(np.sum(acfo_times)),
            "finufft_apply_seconds": float(np.sum(finufft_times)),
            "total_seconds": time.perf_counter() - total_start,
        },
        "rows": rows,
        "technical_gates": gates,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        experimental=experiment,
        finite_mask=finite,
        h_grid=targets["h_grid"],
        k_grid=targets["k_grid"],
        absolute_hkl=targets["absolute_hkl"],
        reduced_hkl=targets["reduced_hkl"],
        candidates=candidates,
        sampled_spins=samples,
        acfo_replica_maps=acfo_replica_maps.reshape(
            candidate_count, replica_count, *experiment.shape
        ),
        finufft_replica_maps=finufft_replica_maps.reshape(
            candidate_count, replica_count, *experiment.shape
        ),
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
