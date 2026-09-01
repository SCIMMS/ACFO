from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from validate_public_waxs_structures import (  # noqa: E402
    build_form_factors,
    load_structure,
)
from waxs_cake import (  # noqa: E402
    PreparedExactCoordinateHarmonicPlan,
    encode_elements,
    repeated_block_translations,
    translation_lattice_factor_separable,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference.ravel())), 1e-300)
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


def debye_waller_amplitude(q_inv_angstrom: np.ndarray, b_iso: float) -> np.ndarray:
    return np.exp(
        -float(b_iso) * np.square(q_inv_angstrom) / (16.0 * np.pi * np.pi)
    )


def score_intensity(candidate: np.ndarray, observed: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(observed.ravel())), 1e-300)
    return float(np.linalg.norm((candidate - observed).ravel()) / denominator)


def add_intensity_noise(
    clean: np.ndarray,
    *,
    relative_l2: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if relative_l2 <= 0.0:
        return np.ascontiguousarray(clean.copy()), {
            "seed": int(seed),
            "target_relative_l2": 0.0,
            "observed_relative_l2": 0.0,
        }
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.shape)
    scale = (
        float(relative_l2)
        * max(float(np.linalg.norm(clean.ravel())), 1e-300)
        / max(float(np.linalg.norm(noise.ravel())), 1e-300)
    )
    observed = np.maximum(clean + scale * noise, 0.0)
    return np.ascontiguousarray(observed), {
        "seed": int(seed),
        "target_relative_l2": float(relative_l2),
        "observed_relative_l2": relative_l2_fn(observed, clean),
        "clipped_negative_pixels": int(np.count_nonzero(clean + scale * noise < 0.0)),
    }


def relative_l2_fn(candidate: np.ndarray, reference: np.ndarray) -> float:
    return relative_l2(candidate, reference)


def candidate_grid(
    *,
    occupancy_min: float,
    occupancy_max: float,
    occupancy_count: int,
    b_min: float,
    b_max: float,
    b_count: int,
) -> list[dict[str, Any]]:
    occupancies = np.linspace(occupancy_min, occupancy_max, occupancy_count)
    b_values = np.linspace(b_min, b_max, b_count)
    rows: list[dict[str, Any]] = []
    state_index = 0
    for occupancy_index, occupancy in enumerate(occupancies):
        for b_index, b_iso in enumerate(b_values):
            rows.append(
                {
                    "state_index": int(state_index),
                    "occupancy_index": int(occupancy_index),
                    "b_index": int(b_index),
                    "subdomain_occupancy": float(occupancy),
                    "b_iso_angstrom2": float(b_iso),
                }
            )
            state_index += 1
    return rows


def top_indices(scores: np.ndarray, count: int) -> list[int]:
    return [
        int(value)
        for value in np.argsort(scores, kind="stable")[: min(count, scores.size)]
    ]


def top_k_overlap(left: list[int], right: list[int]) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "intersection_count": len(intersection),
        "fraction": len(intersection) / max(len(left_set), 1),
        "jaccard": len(intersection) / max(len(union), 1),
        "same_order": left == right,
    }


def timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "total_s": float(array.sum()),
        "median_s": float(np.median(array)),
        "p05_s": float(np.percentile(array, 5)),
        "p95_s": float(np.percentile(array, 95)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def balanced_candidate_order(repeat_index: int, candidate_index: int) -> str:
    """Alternate AB/BA within and across complete candidate-library repeats."""

    return "AB" if (int(repeat_index) + int(candidate_index)) % 2 == 0 else "BA"


def paired_speedup_interval(
    candidate: list[float],
    baseline: list[float],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Paired bootstrap interval for median-baseline / median-candidate time."""

    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size == 0:
        raise ValueError("paired timing samples must be non-empty equal-length vectors")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("paired timing samples must be positive")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(draws.size):
        selected = rng.integers(0, a.size, size=a.size)
        draws[index] = np.median(b[selected]) / np.median(a[selected])
    paired_ratios = b / a
    return {
        "definition": "median(baseline workflow totals) / median(ACFO workflow totals)",
        "point": float(np.median(b) / np.median(a)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "paired_ratio_median": float(np.median(paired_ratios)),
        "paired_ratio_p05": float(np.percentile(paired_ratios, 5)),
        "paired_ratio_p95": float(np.percentile(paired_ratios, 95)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }


def affine_occupancy_basis_weights(occupancy_mask: np.ndarray) -> np.ndarray:
    """Return the exact outside/inside basis for the manufactured library."""

    mask = np.asarray(occupancy_mask, dtype=bool)
    if mask.ndim != 1 or not mask.size:
        raise ValueError("occupancy_mask must be a non-empty one-dimensional array")
    return np.ascontiguousarray(
        np.stack([~mask, mask]).astype(np.complex128, copy=False)
    )


def synthesize_affine_candidate_library(
    basis_amplitudes: np.ndarray,
    occupancies: np.ndarray,
    b_factors: np.ndarray,
) -> np.ndarray:
    """Synthesize every occupancy/B-factor state from two exact FT bases."""

    basis = np.asarray(basis_amplitudes, dtype=np.complex128)
    occupancy = np.asarray(occupancies, dtype=np.float64)
    factors = np.asarray(b_factors, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[0] != 2:
        raise ValueError("basis_amplitudes must have shape (2, n_q, n_phi)")
    if occupancy.ndim != 1:
        raise ValueError("occupancies must be one-dimensional")
    if factors.shape != (occupancy.size, basis.shape[1]):
        raise ValueError("b_factors must have shape (n_state, n_q)")
    combined = basis[0][None, ...] + occupancy[:, None, None] * basis[1][None, ...]
    return combined * factors[:, :, None]


def measure_paired_blocks(
    acfo_execute: Callable[[], np.ndarray],
    baseline_execute: Callable[[], np.ndarray],
    *,
    warmups: int,
    samples: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Measure two complete blocks with balanced ABBA ordering."""

    if warmups < 0 or samples <= 0 or samples % 2:
        raise ValueError("warmups must be non-negative and samples positive/even")
    for index in range(warmups):
        order = ("A", "B") if index % 2 == 0 else ("B", "A")
        for label in order:
            _ = acfo_execute() if label == "A" else baseline_execute()
    acfo_samples: list[float] = []
    baseline_samples: list[float] = []
    for _ in range(samples // 2):
        for label in ("A", "B", "B", "A"):
            gc.collect()
            start = time.perf_counter()
            _ = acfo_execute() if label == "A" else baseline_execute()
            elapsed = time.perf_counter() - start
            (acfo_samples if label == "A" else baseline_samples).append(elapsed)
    return {
        "measurement_unit": "two exact occupancy FT bases plus vectorized 100-state synthesis",
        "ordering": "ABBA",
        "warmup_count": int(warmups),
        "samples_per_arm": int(samples),
        "acfo_seconds": {**timing_summary(acfo_samples), "samples_s": acfo_samples},
        "baseline_seconds": {
            **timing_summary(baseline_samples),
            "samples_s": baseline_samples,
        },
        "baseline_over_acfo_speedup": paired_speedup_interval(
            acfo_samples,
            baseline_samples,
            seed=int(bootstrap_seed),
            bootstrap_samples=int(bootstrap_samples),
        ),
    }


def measure_candidate_library_workflows(
    states: list[dict[str, Any]],
    acfo_execute: Callable[[dict[str, Any]], np.ndarray],
    finufft_execute: Callable[[dict[str, Any]], np.ndarray],
    *,
    warmups: int,
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any] | None:
    """Measure full candidate-library repeats with balanced paired ordering.

    Setup, intensity scoring and observation generation remain outside the
    timed regions, matching the original candidate-evaluation contract.  A
    repeat is the sum of all state-level hot applications for each arm.
    """

    if warmups < 0 or repeats < 0:
        raise ValueError("workflow warmups/repeats must be non-negative")
    if repeats == 0:
        return None
    if not states:
        raise ValueError("states must be non-empty")

    def one_repeat(repeat_index: int, *, measured: bool) -> dict[str, Any]:
        totals = {"acfo": 0.0, "finufft": 0.0}
        calls = {"acfo": 0, "finufft": 0}
        wall_start = time.perf_counter()
        for candidate_index, state in enumerate(states):
            order = balanced_candidate_order(repeat_index, candidate_index)
            for label in order:
                gc.collect()
                start = time.perf_counter()
                if label == "A":
                    _ = acfo_execute(state)
                    key = "acfo"
                else:
                    _ = finufft_execute(state)
                    key = "finufft"
                totals[key] += time.perf_counter() - start
                calls[key] += 1
        row = {
            "repeat_index": int(repeat_index),
            "measured": bool(measured),
            "first_candidate_order": balanced_candidate_order(repeat_index, 0),
            "candidate_count": len(states),
            "acfo_calls": calls["acfo"],
            "finufft_calls": calls["finufft"],
            "acfo_total_s": float(totals["acfo"]),
            "finufft_total_s": float(totals["finufft"]),
            "paired_speedup": float(totals["finufft"] / totals["acfo"]),
            "interleaved_wall_s": float(time.perf_counter() - wall_start),
        }
        stage = "measured" if measured else "warmup"
        print(
            f"workflow {stage} {repeat_index + 1}: "
            f"ACFO={row['acfo_total_s']:.6f}s "
            f"FINUFFT={row['finufft_total_s']:.6f}s "
            f"ratio={row['paired_speedup']:.6f}",
            flush=True,
        )
        return row

    warmup_rows = [
        one_repeat(index, measured=False) for index in range(int(warmups))
    ]
    measured_rows = [
        one_repeat(index + int(warmups), measured=True)
        for index in range(int(repeats))
    ]
    acfo_totals = [float(row["acfo_total_s"]) for row in measured_rows]
    finufft_totals = [float(row["finufft_total_s"]) for row in measured_rows]
    return {
        "measurement_unit": "one complete ordered candidate-library hot-evaluation pass per arm",
        "candidate_count_per_workflow": len(states),
        "scoring_included": False,
        "setup_included": False,
        "observation_generation_included": False,
        "gc_collection_inside_timed_region": False,
        "ordering": "AB/BA alternates by candidate and repeat parity",
        "warmup_count": int(warmups),
        "measured_count": int(repeats),
        "warmup_rows": warmup_rows,
        "measured_rows": measured_rows,
        "acfo_workflow_seconds": {
            **timing_summary(acfo_totals),
            "samples_s": acfo_totals,
        },
        "finufft_workflow_seconds": {
            **timing_summary(finufft_totals),
            "samples_s": finufft_totals,
        },
        "baseline_over_acfo_speedup": paired_speedup_interval(
            acfo_totals,
            finufft_totals,
            seed=int(bootstrap_seed),
            bootstrap_samples=int(bootstrap_samples),
        ),
    }


def build_finufft_plans(
    sources: np.ndarray,
    source_e: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    n_elements: int,
    eps: float,
    threads: int,
    n_trans: int = 1,
) -> tuple[list[Any], list[np.ndarray]]:
    """Prepare one reusable type-3 FINUFFT plan per element channel."""

    import finufft

    if n_trans <= 0:
        raise ValueError("n_trans must be positive")

    plans: list[Any] = []
    masks: list[np.ndarray] = []
    for element_index in range(n_elements):
        mask = np.flatnonzero(source_e == element_index)
        plan = finufft.Plan(
            3,
            3,
            eps=eps,
            isign=1,
            dtype="complex128",
            nthreads=threads,
            n_trans=int(n_trans),
        )
        plan.setpts(
            np.ascontiguousarray(sources[mask, 0]),
            np.ascontiguousarray(sources[mask, 1]),
            np.ascontiguousarray(sources[mask, 2]),
            np.ascontiguousarray(qx),
            np.ascontiguousarray(qy),
            np.ascontiguousarray(qz),
        )
        plans.append(plan)
        masks.append(mask)
    return plans, masks


def execute_finufft_batched_plans(
    plans: list[Any],
    masks: list[np.ndarray],
    source_weights: np.ndarray,
    form_factors: np.ndarray,
    target_q_indices: np.ndarray,
) -> np.ndarray:
    """Execute matched type-3 element plans for a batch of source states."""

    weights = np.asarray(source_weights, dtype=np.complex128)
    if weights.ndim != 2:
        raise ValueError("source_weights must have shape (n_trans, n_source)")
    active = np.zeros((weights.shape[0], target_q_indices.size), dtype=np.complex128)
    for element_index, (plan, mask) in enumerate(zip(plans, masks, strict=True)):
        values = np.asarray(
            plan.execute(np.ascontiguousarray(weights[:, mask])),
            dtype=np.complex128,
        )
        if values.ndim == 1:
            values = values[None, :]
        active += values * form_factors[element_index, target_q_indices][None, :]
    return active


def build_finufft_fused_element_plan(
    sources: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    n_elements: int,
    n_source_bases: int,
    eps: float,
    threads: int,
) -> Any:
    """Prepare one maximum-batching plan over element x source-basis channels."""

    import finufft

    n_trans = int(n_elements) * int(n_source_bases)
    if n_trans <= 0:
        raise ValueError("element and source-basis counts must be positive")
    plan = finufft.Plan(
        3,
        3,
        eps=eps,
        isign=1,
        dtype="complex128",
        nthreads=threads,
        n_trans=n_trans,
    )
    plan.setpts(
        np.ascontiguousarray(sources[:, 0]),
        np.ascontiguousarray(sources[:, 1]),
        np.ascontiguousarray(sources[:, 2]),
        np.ascontiguousarray(qx),
        np.ascontiguousarray(qy),
        np.ascontiguousarray(qz),
    )
    return plan


def execute_finufft_fused_element_plan(
    plan: Any,
    source_e: np.ndarray,
    source_weights: np.ndarray,
    form_factors: np.ndarray,
    target_q_indices: np.ndarray,
) -> np.ndarray:
    """Execute one plan containing every element x source-basis transform."""

    weights = np.asarray(source_weights, dtype=np.complex128)
    elements = np.asarray(source_e, dtype=np.int64)
    if weights.ndim != 2 or elements.shape != (weights.shape[1],):
        raise ValueError("source weights/elements have incompatible shapes")
    n_elements = int(form_factors.shape[0])
    strengths = np.zeros(
        (weights.shape[0], n_elements, weights.shape[1]), dtype=np.complex128
    )
    for element_index in range(n_elements):
        mask = elements == element_index
        strengths[:, element_index, mask] = weights[:, mask]
    values = np.asarray(
        plan.execute(np.ascontiguousarray(strengths.reshape(-1, weights.shape[1]))),
        dtype=np.complex128,
    ).reshape(weights.shape[0], n_elements, -1)
    return np.sum(
        values * form_factors[:, target_q_indices][None, :, :], axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a manufactured occupancy/B-factor WAXS candidate library "
            "and test whether ACFO preserves FINUFFT model ranking."
        )
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path(
            "structures/processed/"
            "protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"
        ),
    )
    parser.add_argument(
        "--supercell",
        type=Path,
        default=Path(
            "structures/processed/"
            "protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz"
        ),
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--comparison-mode",
        choices=["full-supercell", "symmetry-matched"],
        default="full-supercell",
        help=(
            "FINUFFT candidate-evaluation representation.  The legacy "
            "full-supercell mode retains the independent generic workflow; "
            "symmetry-matched gives both timed backends the same unit cell "
            "and exact finite-lattice factor while retaining a full-supercell "
            "FINUFFT observation generator."
        ),
    )
    parser.add_argument("--nq", type=int)
    parser.add_argument("--q-min", type=float, default=6.7)
    parser.add_argument("--q-max", type=float, default=8.0)
    parser.add_argument("--nphi", type=int)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=4)
    parser.add_argument("--occupancy-min", type=float, default=0.7)
    parser.add_argument("--occupancy-max", type=float, default=1.0)
    parser.add_argument("--occupancy-count", type=int)
    parser.add_argument("--b-min", type=float, default=5.0)
    parser.add_argument("--b-max", type=float, default=23.0)
    parser.add_argument("--b-count", type=int)
    parser.add_argument("--truth-occupancy", type=float)
    parser.add_argument("--truth-b", type=float)
    parser.add_argument("--intensity-noise-rel-l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026072611)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--workflow-warmups",
        type=int,
        default=0,
        help="Complete candidate-library warmup passes for publication timing.",
    )
    parser.add_argument(
        "--workflow-repeats",
        type=int,
        default=0,
        help="Complete candidate-library measured passes; zero preserves legacy behavior.",
    )
    parser.add_argument("--workflow-bootstrap-samples", type=int, default=4000)
    parser.add_argument("--workflow-bootstrap-seed", type=int, default=20260819)
    parser.add_argument(
        "--structured-closure-warmups",
        type=int,
        default=0,
        help="Warmups for the exact two-basis occupancy/B-factor library closure.",
    )
    parser.add_argument(
        "--structured-closure-samples",
        type=int,
        default=0,
        help="Balanced samples per arm; zero disables the structured closure.",
    )
    parser.add_argument("--structured-bootstrap-samples", type=int, default=4000)
    parser.add_argument("--structured-bootstrap-seed", type=int, default=20260823)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "waxs_100_state_ranking.json",
    )
    args = parser.parse_args()
    defaults = (
        {
            "nq": 4,
            # Keep the high-q azimuthal Nyquist contract even in smoke mode;
            # reducing only q rows and candidate count makes the test fast
            # without silently switching away from the FFT synthesis path.
            "nphi": 864,
            "occupancy_count": 3,
            "b_count": 3,
            "truth_occupancy": 0.86,
            "truth_b": 13.4,
        }
        if args.mode == "smoke"
        else {
            "nq": 16,
            "nphi": 864,
            "occupancy_count": 10,
            "b_count": 10,
            # The observation is deliberately off-grid so the ranking test is
            # not reduced to reproducing a candidate that generated itself.
            "truth_occupancy": 0.915,
            "truth_b": 12.4,
        }
    )
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.nphi % 2 or args.nphi < 4:
        raise ValueError("nphi must be an even integer >= 4")
    if args.workflow_warmups < 0 or args.workflow_repeats < 0:
        raise ValueError("workflow warmups/repeats must be non-negative")
    if args.structured_closure_warmups < 0 or args.structured_closure_samples < 0:
        raise ValueError("structured closure warmups/samples must be non-negative")
    if args.structured_closure_samples % 2:
        raise ValueError("structured closure samples must be even")
    if args.workflow_repeats and args.comparison_mode != "symmetry-matched":
        raise ValueError(
            "publication workflow timing requires --comparison-mode symmetry-matched"
        )
    if args.structured_closure_samples and args.comparison_mode != "symmetry-matched":
        raise ValueError(
            "structured library closure requires --comparison-mode symmetry-matched"
        )
    states = candidate_grid(
        occupancy_min=args.occupancy_min,
        occupancy_max=args.occupancy_max,
        occupancy_count=args.occupancy_count,
        b_min=args.b_min,
        b_max=args.b_max,
        b_count=args.b_count,
    )
    truth_state = {
        "state_index": None,
        "occupancy_index": None,
        "b_index": None,
        "subdomain_occupancy": float(args.truth_occupancy),
        "b_iso_angstrom2": float(args.truth_b),
        "off_grid": True,
    }

    unit_path = args.unit if args.unit.is_absolute() else ROOT / args.unit
    supercell_path = (
        args.supercell if args.supercell.is_absolute() else ROOT / args.supercell
    )
    unit_coords, unit_elements, unit_metadata = load_structure(unit_path)
    coords, elements, metadata = load_structure(supercell_path)
    blocks = elements.reshape(-1, unit_elements.size)
    if not np.array_equal(blocks, np.broadcast_to(unit_elements, blocks.shape)):
        raise RuntimeError("supercell does not repeat the unit element ordering")
    translations, repetition_residual = repeated_block_translations(
        unit_coords,
        coords,
        atol=1e-9,
    )
    unit_e, element_order = encode_elements(unit_elements)
    element_indices, _ = encode_elements(elements, element_order=element_order)
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)
    qx = np.ascontiguousarray(
        (q_perp[:, None] * np.cos(phi)[None, :]).ravel()
    )
    qy = np.ascontiguousarray(
        (q_perp[:, None] * np.sin(phi)[None, :]).ravel()
    )
    qz = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], (args.nq, args.nphi)).ravel()
    )
    target_q_indices = np.repeat(np.arange(args.nq), args.nphi)
    ff_mapping = build_form_factors(unit_elements, q_solver, "xray_f0")
    form_factors = normalize_form_factors(
        element_order,
        q_solver,
        ff_mapping,
    ).astype(np.complex128, copy=False)
    # A deterministic coordinate-defined subdomain is used because the frozen
    # processed NPZ stores coordinates/elements but no residue identifiers.
    center = np.median(unit_coords, axis=0)
    distances = np.linalg.norm(unit_coords - center[None, :], axis=1)
    threshold = float(np.quantile(distances, 0.25))
    occupancy_mask = distances <= threshold
    if not np.any(occupancy_mask):
        raise RuntimeError("manufactured occupancy subdomain is empty")

    setup_start = time.perf_counter()
    lattice = translation_lattice_factor_separable(
        qx,
        qy,
        qz,
        translations,
        metadata["supercell"],
    ).reshape(args.nq, args.nphi)
    prepared = PreparedExactCoordinateHarmonicPlan(
        unit_coords,
        q_perp,
        q_z_rows,
        phi,
        element_indices=unit_e,
        form_factors=form_factors,
        harmonic_margin=args.harmonic_margin,
        prepare_direct_basis=False,
        coefficient_backend="fused_phase",
    )
    if not prepared.fft_supported:
        raise RuntimeError("prepared ACFO plan does not support FFT synthesis")
    acfo_setup_s = time.perf_counter() - setup_start
    def weights_for_state(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        unit_weights = np.ones(unit_coords.shape[0], dtype=np.complex128)
        unit_weights[occupancy_mask] = float(state["subdomain_occupancy"])
        source_weights = np.tile(unit_weights, int(translations.shape[0]))
        if source_weights.shape[0] != coords.shape[0]:
            raise RuntimeError("repeated state weights do not match supercell atoms")
        return unit_weights, source_weights

    def apply_b_factor(amplitude: np.ndarray, b_iso: float) -> np.ndarray:
        factor = debye_waller_amplitude(q_report, b_iso)
        return amplitude * factor[:, None]

    def acfo_execute(state: dict[str, Any]) -> np.ndarray:
        unit_weights, _ = weights_for_state(state)
        unit, _ = prepared.execute(
            atom_weights=unit_weights,
            synthesis_backend="fft",
        )
        return apply_b_factor(unit * lattice, state["b_iso_angstrom2"])

    def execute_finufft_plans(
        local_plans: list[Any],
        local_masks: list[np.ndarray],
        source_weights: np.ndarray,
        *,
        apply_lattice_factor: bool,
        b_iso: float,
    ) -> np.ndarray:
        active = np.zeros(qx.size, dtype=np.complex128)
        for element_index, (plan, mask) in enumerate(
            zip(local_plans, local_masks, strict=True)
        ):
            values = plan.execute(np.ascontiguousarray(source_weights[mask]))
            active += values * form_factors[element_index, target_q_indices]
        amplitude = active.reshape(args.nq, args.nphi)
        if apply_lattice_factor:
            amplitude = amplitude * lattice
        return apply_b_factor(amplitude, b_iso)

    reference_setup_start = time.perf_counter()
    reference_plans, reference_masks = build_finufft_plans(
        coords,
        element_indices,
        qx,
        qy,
        qz,
        n_elements=len(element_order),
        eps=args.finufft_eps,
        threads=args.finufft_threads,
    )
    reference_setup_s = time.perf_counter() - reference_setup_start

    _, truth_supercell_weights = weights_for_state(truth_state)
    reference_execute_start = time.perf_counter()
    truth_amplitude = execute_finufft_plans(
        reference_plans,
        reference_masks,
        truth_supercell_weights,
        apply_lattice_factor=False,
        b_iso=float(truth_state["b_iso_angstrom2"]),
    )
    reference_execute_s = time.perf_counter() - reference_execute_start

    if args.comparison_mode == "full-supercell":
        plans = reference_plans
        masks = reference_masks
        finufft_setup_s = reference_setup_s
        finufft_source = "full repeated supercell"
        finufft_source_count = int(coords.shape[0])
    else:
        del reference_plans, reference_masks
        gc.collect()
        setup_start = time.perf_counter()
        plans, masks = build_finufft_plans(
            unit_coords,
            unit_e,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            eps=args.finufft_eps,
            threads=args.finufft_threads,
        )
        finufft_setup_s = time.perf_counter() - setup_start
        finufft_source = "unit cell plus shared exact finite-lattice factor"
        finufft_source_count = int(unit_coords.shape[0])

    def finufft_execute(state: dict[str, Any]) -> np.ndarray:
        unit_weights, source_weights = weights_for_state(state)
        matched = args.comparison_mode == "symmetry-matched"
        return execute_finufft_plans(
            plans,
            masks,
            unit_weights if matched else source_weights,
            apply_lattice_factor=matched,
            b_iso=float(state["b_iso_angstrom2"]),
        )

    # The observed map is generated with the generic full-supercell baseline.
    clean_intensity = np.square(np.abs(truth_amplitude))
    observed_intensity, noise_metadata = add_intensity_noise(
        clean_intensity,
        relative_l2=args.intensity_noise_rel_l2,
        seed=args.seed,
    )
    acfo_execute(states[0])
    finufft_execute(states[0])
    rows: list[dict[str, Any]] = []
    acfo_times: list[float] = []
    finufft_times: list[float] = []
    for candidate_index, state in enumerate(states):
        order = "AB" if candidate_index % 2 == 0 else "BA"
        outputs: dict[str, np.ndarray] = {}
        times: dict[str, float] = {}
        for label in order:
            gc.collect()
            start = time.perf_counter()
            if label == "A":
                outputs["acfo"] = acfo_execute(state)
                times["acfo"] = time.perf_counter() - start
            else:
                outputs["finufft"] = finufft_execute(state)
                times["finufft"] = time.perf_counter() - start
        acfo_times.append(times["acfo"])
        finufft_times.append(times["finufft"])
        acfo_intensity = np.square(np.abs(outputs["acfo"]))
        finufft_intensity = np.square(np.abs(outputs["finufft"]))
        row = {
            **state,
            "order": order,
            "acfo_s": times["acfo"],
            "finufft_s": times["finufft"],
            "acfo_score": score_intensity(acfo_intensity, observed_intensity),
            "finufft_score": score_intensity(
                finufft_intensity,
                observed_intensity,
            ),
            "complex_relative_l2": relative_l2(
                outputs["acfo"],
                outputs["finufft"],
            ),
            "intensity_relative_l2": relative_l2(
                acfo_intensity,
                finufft_intensity,
            ),
        }
        rows.append(row)
        print(
            f"state {candidate_index + 1}/{len(states)} "
            f"{order}: ACFO={times['acfo']:.6f}s "
            f"FINUFFT={times['finufft']:.6f}s "
            f"scores={row['acfo_score']:.6g}/{row['finufft_score']:.6g}",
            flush=True,
        )
    acfo_scores = np.asarray([row["acfo_score"] for row in rows])
    finufft_scores = np.asarray([row["finufft_score"] for row in rows])
    acfo_top = top_indices(acfo_scores, args.top_k)
    finufft_top = top_indices(finufft_scores, args.top_k)
    rank_correlation = float(spearmanr(acfo_scores, finufft_scores).statistic)
    overlap = top_k_overlap(acfo_top, finufft_top)
    occupancy_span = max(float(args.occupancy_max - args.occupancy_min), 1e-300)
    b_span = max(float(args.b_max - args.b_min), 1e-300)

    def normalized_parameter_distance(state: dict[str, Any]) -> float:
        return float(
            np.hypot(
                (
                    float(state["subdomain_occupancy"])
                    - float(truth_state["subdomain_occupancy"])
                )
                / occupancy_span,
                (
                    float(state["b_iso_angstrom2"])
                    - float(truth_state["b_iso_angstrom2"])
                )
                / b_span,
            )
        )

    max_complex_l2 = max(float(row["complex_relative_l2"]) for row in rows)
    max_intensity_l2 = max(float(row["intensity_relative_l2"]) for row in rows)
    acfo_timing = timing_summary(acfo_times)
    finufft_timing = timing_summary(finufft_times)
    publication_workflow_timing = measure_candidate_library_workflows(
        states,
        acfo_execute,
        finufft_execute,
        warmups=int(args.workflow_warmups),
        repeats=int(args.workflow_repeats),
        bootstrap_samples=int(args.workflow_bootstrap_samples),
        bootstrap_seed=int(args.workflow_bootstrap_seed),
    )
    structured_library_closure = None
    if args.structured_closure_samples:
        basis_weights = affine_occupancy_basis_weights(occupancy_mask)
        occupancies = np.asarray(
            [float(state["subdomain_occupancy"]) for state in states],
            dtype=np.float64,
        )
        structured_b_factors = np.stack(
            [
                debye_waller_amplitude(q_report, float(state["b_iso_angstrom2"]))
                for state in states
            ]
        )

        structured_setup_start = time.perf_counter()
        structured_plans, structured_masks = build_finufft_plans(
            unit_coords,
            unit_e,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            eps=args.finufft_eps,
            threads=args.finufft_threads,
            n_trans=2,
        )
        element_plan_setup_s = time.perf_counter() - structured_setup_start
        fused_setup_start = time.perf_counter()
        fused_structured_plan = build_finufft_fused_element_plan(
            unit_coords,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            n_source_bases=2,
            eps=args.finufft_eps,
            threads=args.finufft_threads,
        )
        fused_plan_setup_s = time.perf_counter() - fused_setup_start

        def acfo_structured_execute() -> np.ndarray:
            bases = []
            for weights in basis_weights:
                unit_basis, _ = prepared.execute(
                    atom_weights=weights,
                    synthesis_backend="fft",
                )
                bases.append(unit_basis * lattice)
            return synthesize_affine_candidate_library(
                np.stack(bases), occupancies, structured_b_factors
            )

        def finufft_structured_execute() -> np.ndarray:
            flat = execute_finufft_batched_plans(
                structured_plans,
                structured_masks,
                basis_weights,
                form_factors,
                target_q_indices,
            )
            bases = flat.reshape(2, args.nq, args.nphi) * lattice[None, ...]
            return synthesize_affine_candidate_library(
                bases, occupancies, structured_b_factors
            )

        def finufft_fused_structured_execute() -> np.ndarray:
            flat = execute_finufft_fused_element_plan(
                fused_structured_plan,
                unit_e,
                basis_weights,
                form_factors,
                target_q_indices,
            )
            bases = flat.reshape(2, args.nq, args.nphi) * lattice[None, ...]
            return synthesize_affine_candidate_library(
                bases, occupancies, structured_b_factors
            )

        acfo_structured_value = acfo_structured_execute()
        finufft_element_value = finufft_structured_execute()
        finufft_fused_value = finufft_fused_structured_execute()
        element_fused_relative_l2 = relative_l2(
            finufft_element_value, finufft_fused_value
        )
        element_per_state_complex = [
            relative_l2(acfo_structured_value[index], finufft_element_value[index])
            for index in range(len(states))
        ]
        element_per_state_intensity = [
            relative_l2(
                np.square(np.abs(acfo_structured_value[index])),
                np.square(np.abs(finufft_element_value[index])),
            )
            for index in range(len(states))
        ]
        fused_per_state_complex = [
            relative_l2(acfo_structured_value[index], finufft_fused_value[index])
            for index in range(len(states))
        ]
        fused_per_state_intensity = [
            relative_l2(
                np.square(np.abs(acfo_structured_value[index])),
                np.square(np.abs(finufft_fused_value[index])),
            )
            for index in range(len(states))
        ]
        structured_acfo_scores = np.asarray(
            [
                score_intensity(np.square(np.abs(value)), observed_intensity)
                for value in acfo_structured_value
            ]
        )
        element_finufft_scores = np.asarray(
            [
                score_intensity(np.square(np.abs(value)), observed_intensity)
                for value in finufft_element_value
            ]
        )
        fused_finufft_scores = np.asarray(
            [
                score_intensity(np.square(np.abs(value)), observed_intensity)
                for value in finufft_fused_value
            ]
        )
        element_plan_timing = measure_paired_blocks(
            acfo_structured_execute,
            finufft_structured_execute,
            warmups=int(args.structured_closure_warmups),
            samples=int(args.structured_closure_samples),
            bootstrap_samples=int(args.structured_bootstrap_samples),
            bootstrap_seed=int(args.structured_bootstrap_seed),
        )
        fused_plan_timing = measure_paired_blocks(
            acfo_structured_execute,
            finufft_fused_structured_execute,
            warmups=int(args.structured_closure_warmups),
            samples=int(args.structured_closure_samples),
            bootstrap_samples=int(args.structured_bootstrap_samples),
            bootstrap_seed=int(args.structured_bootstrap_seed) + 1,
        )
        baseline_variants = {
            "four_element_plans_n_trans_2": {
                "plan_count": len(element_order),
                "n_trans_per_plan": 2,
                "total_fourier_channels": 2 * len(element_order),
                "zero_padded_source_channels": False,
                "setup_s": float(element_plan_setup_s),
                "timing": element_plan_timing,
            },
            "single_plan_n_trans_8": {
                "plan_count": 1,
                "n_trans_per_plan": 2 * len(element_order),
                "total_fourier_channels": 2 * len(element_order),
                "zero_padded_source_channels": True,
                "setup_s": float(fused_plan_setup_s),
                "timing": fused_plan_timing,
            },
        }
        selected_baseline_id = min(
            baseline_variants,
            key=lambda name: baseline_variants[name]["timing"][
                "baseline_seconds"
            ]["median_s"],
        )
        structured_timing = baseline_variants[selected_baseline_id]["timing"]
        accuracy_variants = {
            "four_element_plans_n_trans_2": {
                "maximum_complex_relative_l2": float(
                    max(element_per_state_complex)
                ),
                "maximum_intensity_relative_l2": float(
                    max(element_per_state_intensity)
                ),
                "same_top1": bool(
                    int(np.argmin(structured_acfo_scores))
                    == int(np.argmin(element_finufft_scores))
                ),
                "score_spearman": float(
                    spearmanr(structured_acfo_scores, element_finufft_scores).statistic
                ),
            },
            "single_plan_n_trans_8": {
                "maximum_complex_relative_l2": float(
                    max(fused_per_state_complex)
                ),
                "maximum_intensity_relative_l2": float(
                    max(fused_per_state_intensity)
                ),
                "same_top1": bool(
                    int(np.argmin(structured_acfo_scores))
                    == int(np.argmin(fused_finufft_scores))
                ),
                "score_spearman": float(
                    spearmanr(structured_acfo_scores, fused_finufft_scores).statistic
                ),
            },
        }
        selected_accuracy = dict(accuracy_variants[selected_baseline_id])
        selected_accuracy["element_plan_vs_fused_plan_relative_l2"] = float(
            element_fused_relative_l2
        )
        structured_library_closure = {
            "schema": "waxs-affine-library-closure-v1",
            "contract": {
                "candidate_count": len(states),
                "unique_occupancies": int(args.occupancy_count),
                "unique_b_factors": int(args.b_count),
                "source_basis_count": 2,
                "element_channel_count": len(element_order),
                "total_element_basis_fourier_channels": 2 * len(element_order),
                "source_identity": "w(o) = outside_mask + o * inside_mask",
                "b_factor_application": "post-FT vectorized amplitude scaling",
                "same_unit_cell_and_exact_lattice_factor": True,
                "scoring_included_in_timing": False,
                "setup_included_in_timing": False,
            },
            "setup": {
                "acfo_geometry_setup_s": float(acfo_setup_s),
                "finufft_plan_type": 3,
                "tested_finufft_batching_variants": [
                    "four element-specific plans with n_trans=2",
                    f"one shared-source plan with n_trans={2 * len(element_order)}",
                ],
            },
            "accuracy": selected_accuracy,
            "accuracy_variants": accuracy_variants,
            "baseline_variants": baseline_variants,
            "selected_fastest_tested_baseline": selected_baseline_id,
            "timing": structured_timing,
            "claim_boundary": [
                "This is the strongest tested task-level implementation for the frozen affine occupancy/B-factor library.",
                "Both arms use two occupancy source bases and retain four element channels for q-dependent form factors.",
                "FINUFFT tests both element-specific n_trans=2 plans and one maximum-batching n_trans=8 shared-source plan; the faster measured variant defines the reported comparator.",
                "The earlier 100-call row remains an independent-state operator-throughput contract, not the strongest implementation of this structured library.",
            ],
        }
    payload = {
        "schema": (
            "waxs-100-state-publication-timing-v1"
            if publication_workflow_timing is not None
            else "waxs-100-state-ranking-v2"
        ),
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "comparison_mode": args.comparison_mode,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "finufft_threads": int(args.finufft_threads),
        },
        "structures": {
            "unit_path": unit_path.relative_to(ROOT).as_posix(),
            "unit_sha256": sha256(unit_path),
            "unit_structure_id": unit_metadata.get("structure_id"),
            "supercell_path": supercell_path.relative_to(ROOT).as_posix(),
            "supercell_sha256": sha256(supercell_path),
            "supercell_shape": metadata.get("supercell"),
            "unit_atom_count": int(unit_coords.shape[0]),
            "supercell_atom_count": int(coords.shape[0]),
            "cell_count": int(translations.shape[0]),
            "repetition_residual_nm": float(repetition_residual),
        },
        "problem": {
            "candidate_count": len(states),
            "candidate_axes": {
                "subdomain_occupancy": [
                    float(args.occupancy_min),
                    float(args.occupancy_max),
                    int(args.occupancy_count),
                ],
                "b_iso_angstrom2": [
                    float(args.b_min),
                    float(args.b_max),
                    int(args.b_count),
                ],
            },
            "occupancy_subdomain": {
                "definition": "unit-cell atoms within the first radial-distance quartile from the coordinate-wise median",
                "atom_count": int(np.count_nonzero(occupancy_mask)),
                "fraction": float(np.mean(occupancy_mask)),
                "manufactured_parameter_set": True,
            },
            "truth_state": truth_state,
            "q_range_inv_angstrom": [float(args.q_min), float(args.q_max)],
            "nq": int(args.nq),
            "nphi": int(args.nphi),
            "target_count": int(args.nq * args.nphi),
            "wavelength_nm": float(args.wavelength_nm),
            "observed_intensity_sha256": array_sha256(observed_intensity),
            "noise": noise_metadata,
            "observation_generator": {
                "backend": "FINUFFT type-3",
                "source": "full repeated supercell",
                "source_count": int(coords.shape[0]),
                "setup_s": float(reference_setup_s),
                "execute_s": float(reference_execute_s),
                "excluded_from_candidate_timing": True,
            },
        },
        "operators": {
            "acfo": {
                "prepared_unit_cell": True,
                "finite_lattice_factor": "separable exact translation factor",
                "harmonic_margin": int(args.harmonic_margin),
                "setup_s": float(acfo_setup_s),
                "timing": acfo_timing,
            },
            "finufft": {
                "source": finufft_source,
                "source_count": finufft_source_count,
                "shared_finite_lattice_factor": bool(
                    args.comparison_mode == "symmetry-matched"
                ),
                "eps": float(args.finufft_eps),
                "threads": int(args.finufft_threads),
                "setup_s": float(finufft_setup_s),
                "timing": finufft_timing,
            },
        },
        "ranking": {
            "spearman": rank_correlation,
            "top_k": int(args.top_k),
            "acfo_top_indices": acfo_top,
            "finufft_top_indices": finufft_top,
            "top_k_overlap": overlap,
            "acfo_best_state": states[acfo_top[0]],
            "finufft_best_state": states[finufft_top[0]],
            "off_grid_truth": truth_state,
            "acfo_best_normalized_parameter_distance": normalized_parameter_distance(
                states[acfo_top[0]]
            ),
            "finufft_best_normalized_parameter_distance": normalized_parameter_distance(
                states[finufft_top[0]]
            ),
        },
        "accuracy": {
            "maximum_complex_relative_l2": max_complex_l2,
            "maximum_intensity_relative_l2": max_intensity_l2,
        },
        "publication_workflow_timing": publication_workflow_timing,
        "structured_library_closure": structured_library_closure,
        "outcome_targets": {
            "spearman_ge_0p999": bool(rank_correlation >= 0.999),
            "same_top1": acfo_top[0] == finufft_top[0],
            "top5_jaccard_eq_1": bool(
                args.top_k != 5 or math.isclose(overlap["jaccard"], 1.0)
            ),
            "maximum_intensity_relative_l2_le_5e_6": bool(
                max_intensity_l2 <= 5e-6
            ),
        },
        "rows": rows,
        "claim_boundary": [
            "This is a manufactured occupancy/B-factor candidate library, not a fitted crystallographic ensemble.",
            "The coordinate-defined occupancy subdomain is used because the frozen processed NPZ does not retain residue identifiers.",
            "The finite lattice factor is a standard crystallographic specialization and is not claimed as an ACFO novelty.",
            (
                "Both timed backends use the same unit cell and exact finite-lattice factor."
                if args.comparison_mode == "symmetry-matched"
                else "ACFO uses a prepared unit-cell plus exact lattice-factor workflow; FINUFFT evaluates the full repeated supercell."
            ),
            "The independent noisy observation is generated by full-supercell FINUFFT and excluded from candidate-evaluation timing.",
            (
                "The timed comparison is representation matched."
                if args.comparison_mode == "symmetry-matched"
                else "The primary scientific claim is preservation of model ranking and top-k decisions, not an operator-only speedup."
            ),
            (
                "Publication timing uses complete candidate-library repeats with setup, scoring and observation generation excluded."
                if publication_workflow_timing is not None
                else "The single-pass per-state timing is descriptive rather than a publication timing distribution."
            ),
            "Anomalous scattering, solvent/background models, orientation uncertainty, and experimental detector noise are not included.",
        ],
    }
    write_json(args.out, payload)
    write_csv(args.out.with_suffix(".csv"), rows)
    lines = [
        "# WAXS 100-state ranking",
        "",
        f"- candidates: `{len(states)}`",
        f"- Spearman score correlation: `{rank_correlation:.9f}`",
        f"- ACFO / FINUFFT top-1: `{acfo_top[0]} / {finufft_top[0]}`",
        f"- top-{args.top_k} Jaccard: `{overlap['jaccard']:.6f}`",
        f"- maximum intensity rel-L2: `{max_intensity_l2:.6g}`",
        f"- ACFO candidate-library total: `{acfo_timing['total_s']:.6f}` s",
        f"- FINUFFT candidate-library total: `{finufft_timing['total_s']:.6f}` s",
        "",
        "The observation is generated independently with full-supercell FINUFFT. "
        + (
            "Timed candidate evaluation gives both backends the same unit cell "
            "and exact finite-lattice factor."
            if args.comparison_mode == "symmetry-matched"
            else "Timed candidate evaluation retains the declared asymmetric structure-aware workflows."
        ),
    ]
    if publication_workflow_timing is not None:
        speed = publication_workflow_timing["baseline_over_acfo_speedup"]
        lines.extend(
            [
                "",
                "## Publication workflow timing",
                "",
                f"- warmup / measured workflows: `{args.workflow_warmups} / {args.workflow_repeats}`",
                f"- FINUFFT/ACFO point ratio: `{speed['point']:.6f}`",
                f"- paired bootstrap 95% interval: `[{speed['lower_95']:.6f}, {speed['upper_95']:.6f}]`",
                "- setup, scoring and observation generation are excluded from each timed workflow",
            ]
        )
    if structured_library_closure is not None:
        speed = structured_library_closure["timing"]["baseline_over_acfo_speedup"]
        accuracy = structured_library_closure["accuracy"]
        lines.extend(
            [
                "",
                "## Task-optimal affine-library closure",
                "",
                "- both arms evaluate two exact occupancy source bases",
                "- FINUFFT uses one `n_trans=2` type-3 batch per element channel",
                f"- FINUFFT/ACFO point ratio: `{speed['point']:.6f}`",
                f"- paired bootstrap 95% interval: `[{speed['lower_95']:.6f}, {speed['upper_95']:.6f}]`",
                f"- maximum complex rel-L2: `{accuracy['maximum_complex_relative_l2']:.6g}`",
                "- B-factor scaling, but not ranking/scoring, is included in the timed block",
            ]
        )
    args.out.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidate_count": len(states),
                "ranking": payload["ranking"],
                "accuracy": payload["accuracy"],
                "outcome_targets": payload["outcome_targets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
