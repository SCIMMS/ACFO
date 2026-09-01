from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_waxs_100_state_ranking import (  # noqa: E402
    add_intensity_noise,
    affine_occupancy_basis_weights,
    array_sha256,
    build_finufft_fused_element_plan,
    build_finufft_plans,
    candidate_grid,
    debye_waller_amplitude,
    execute_finufft_batched_plans,
    execute_finufft_fused_element_plan,
    measure_paired_blocks,
    relative_l2,
    score_intensity,
    synthesize_affine_candidate_library,
    top_indices,
    top_k_overlap,
    write_csv,
    write_json,
)
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import (  # noqa: E402
    PreparedExactCoordinateHarmonicPlan,
    encode_elements,
    repeated_block_translations,
    translation_lattice_factor_separable,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "WAXS_100_STATE_EPS_FRONTIER_PROTOCOL.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_epsilon_candidates(values: list[float]) -> list[float]:
    eps = [float(value) for value in values]
    if not eps or any(not math.isfinite(value) or value <= 0.0 for value in eps):
        raise ValueError("epsilon candidates must be finite and positive")
    if any(left <= right for left, right in zip(eps, eps[1:], strict=False)):
        raise ValueError("epsilon candidates must be strictly loose-to-tight")
    return eps


def select_loosest_passing_epsilon(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first passing row; caller preserves the frozen loose-to-tight order."""

    for row in rows:
        if row.get("accuracy_pass") is True:
            return row
    return None


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else -1.0


def library_accuracy_metrics(
    acfo: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    observed_intensity: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Return accuracy/ranking evidence only; this routine does not time any arm."""

    left = np.asarray(acfo, dtype=np.complex128)
    trial = np.asarray(candidate, dtype=np.complex128)
    truth = np.asarray(reference, dtype=np.complex128)
    if left.shape != trial.shape or left.shape != truth.shape or left.ndim != 3:
        raise ValueError("library arrays must share shape (n_state, n_q, n_phi)")

    left_intensity = np.square(np.abs(left))
    trial_intensity = np.square(np.abs(trial))
    truth_intensity = np.square(np.abs(truth))
    acfo_scores = np.asarray(
        [score_intensity(value, observed_intensity) for value in left_intensity]
    )
    candidate_scores = np.asarray(
        [score_intensity(value, observed_intensity) for value in trial_intensity]
    )
    reference_scores = np.asarray(
        [score_intensity(value, observed_intensity) for value in truth_intensity]
    )
    acfo_top = top_indices(acfo_scores, top_k)
    candidate_top = top_indices(candidate_scores, top_k)
    reference_top = top_indices(reference_scores, top_k)

    def maximum_state_error(values: np.ndarray, target: np.ndarray) -> float:
        return float(max(relative_l2(values[i], target[i]) for i in range(values.shape[0])))

    return {
        "maximum_acfo_vs_candidate_complex_relative_l2": maximum_state_error(
            left, trial
        ),
        "maximum_acfo_vs_candidate_intensity_relative_l2": maximum_state_error(
            left_intensity, trial_intensity
        ),
        "maximum_candidate_vs_reference_complex_relative_l2": maximum_state_error(
            trial, truth
        ),
        "maximum_candidate_vs_reference_intensity_relative_l2": maximum_state_error(
            trial_intensity, truth_intensity
        ),
        "maximum_acfo_vs_reference_complex_relative_l2": maximum_state_error(
            left, truth
        ),
        "maximum_acfo_vs_reference_intensity_relative_l2": maximum_state_error(
            left_intensity, truth_intensity
        ),
        "acfo_vs_candidate_ranking": {
            "same_top1": bool(acfo_top[0] == candidate_top[0]),
            "top_k": int(min(top_k, left.shape[0])),
            "top_k_overlap": top_k_overlap(acfo_top, candidate_top),
            "spearman": _finite_spearman(acfo_scores, candidate_scores),
            "acfo_top_indices": acfo_top,
            "candidate_top_indices": candidate_top,
        },
        "candidate_vs_reference_ranking": {
            "same_top1": bool(candidate_top[0] == reference_top[0]),
            "top_k_overlap": top_k_overlap(candidate_top, reference_top),
            "spearman": _finite_spearman(candidate_scores, reference_scores),
            "reference_top_indices": reference_top,
        },
    }


def accuracy_gates(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    ranking = metrics["acfo_vs_candidate_ranking"]
    return {
        "acfo_vs_candidate_intensity": float(
            metrics["maximum_acfo_vs_candidate_intensity_relative_l2"]
        )
        <= float(gates["maximum_acfo_vs_candidate_intensity_relative_l2"]),
        "candidate_vs_reference_intensity": float(
            metrics["maximum_candidate_vs_reference_intensity_relative_l2"]
        )
        <= float(gates["maximum_candidate_vs_reference_intensity_relative_l2"]),
        "acfo_vs_reference_intensity": float(
            metrics["maximum_acfo_vs_reference_intensity_relative_l2"]
        )
        <= float(gates["maximum_acfo_vs_reference_intensity_relative_l2"]),
        "same_top1": bool(ranking["same_top1"]) == bool(gates["same_top1"]),
        "top_k_jaccard": float(ranking["top_k_overlap"]["jaccard"])
        >= float(gates["top_k_jaccard_min"]),
        "spearman": float(ranking["spearman"]) >= float(gates["spearman_min"]),
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _profile(protocol: dict[str, Any], mode: str) -> dict[str, int]:
    problem = protocol["problem"]
    timing = protocol["final_timing"]
    if mode == "full":
        return {
            "nq": int(problem["nq"]),
            "nphi": int(problem["nphi"]),
            "occupancy_count": int(problem["occupancy_count"]),
            "b_count": int(problem["b_count"]),
            "warmups": int(timing["warmups"]),
            "samples": int(timing["samples_per_arm"]),
            "bootstrap_samples": int(timing["bootstrap_samples"]),
        }
    smoke = protocol["smoke_overrides"]
    return {
        "nq": int(smoke["nq"]),
        "nphi": int(problem["nphi"]),
        "occupancy_count": int(smoke["occupancy_count"]),
        "b_count": int(smoke["b_count"]),
        "warmups": int(smoke["warmups"]),
        "samples": int(smoke["samples_per_arm"]),
        "bootstrap_samples": int(smoke["bootstrap_samples"]),
    }


def run(protocol_path: Path, *, mode: str, output: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    if protocol.get("schema") != "waxs-affine-library-eps-frontier-protocol-v1":
        raise ValueError("unexpected WAXS epsilon-frontier protocol schema")
    problem = protocol["problem"]
    acfo_contract = protocol["acfo"]
    finufft_contract = protocol["finufft"]
    selection_contract = protocol["accuracy_selection"]
    profile = _profile(protocol, mode)
    eps_candidates = validate_epsilon_candidates(
        finufft_contract["epsilon_candidates_loose_to_tight"]
    )
    if int(finufft_contract["n_trans"]) != 8:
        raise ValueError("the frozen fused comparator must use n_trans=8")
    if bool(selection_contract["uses_timing"]):
        raise ValueError("epsilon selection must not use timing")
    if profile["samples"] <= 0 or profile["samples"] % 2:
        raise ValueError("timing samples must be positive and even")

    unit_path = _resolve(ROOT, problem["unit_structure"])
    supercell_path = _resolve(ROOT, problem["supercell_structure"])
    unit_coords, unit_elements, unit_metadata = load_structure(unit_path)
    super_coords, super_elements, super_metadata = load_structure(supercell_path)
    repeated_elements = super_elements.reshape(-1, unit_elements.size)
    if not np.array_equal(
        repeated_elements, np.broadcast_to(unit_elements, repeated_elements.shape)
    ):
        raise RuntimeError("supercell does not repeat the unit element ordering")
    translations, repetition_residual = repeated_block_translations(
        unit_coords, super_coords, atol=1e-9
    )
    unit_e, element_order = encode_elements(unit_elements)
    super_e, _ = encode_elements(super_elements, element_order=element_order)
    if len(element_order) * int(finufft_contract["source_basis_count"]) != 8:
        raise RuntimeError("the data do not realize the frozen n_trans=8 contract")

    q_report = np.linspace(
        float(problem["q_range_inv_angstrom"][0]),
        float(problem["q_range_inv_angstrom"][1]),
        profile["nq"],
        dtype=np.float64,
    )
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, float(problem["wavelength_nm"]))
    phi = (np.arange(profile["nphi"]) + 0.5) * (
        2.0 * np.pi / profile["nphi"]
    )
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :]).ravel())
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :]).ravel())
    qz = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], (profile["nq"], profile["nphi"])).ravel()
    )
    target_q_indices = np.repeat(np.arange(profile["nq"]), profile["nphi"])
    form_factors = normalize_form_factors(
        element_order,
        q_solver,
        build_form_factors(unit_elements, q_solver, "xray_f0"),
    ).astype(np.complex128, copy=False)

    center = np.median(unit_coords, axis=0)
    radii_from_center = np.linalg.norm(unit_coords - center[None, :], axis=1)
    occupancy_mask = radii_from_center <= float(np.quantile(radii_from_center, 0.25))
    basis_weights = affine_occupancy_basis_weights(occupancy_mask)
    states = candidate_grid(
        occupancy_min=float(problem["occupancy_range"][0]),
        occupancy_max=float(problem["occupancy_range"][1]),
        occupancy_count=profile["occupancy_count"],
        b_min=float(problem["b_iso_range_angstrom2"][0]),
        b_max=float(problem["b_iso_range_angstrom2"][1]),
        b_count=profile["b_count"],
    )
    occupancies = np.asarray(
        [float(row["subdomain_occupancy"]) for row in states], dtype=np.float64
    )
    b_factors = np.stack(
        [
            debye_waller_amplitude(q_report, float(row["b_iso_angstrom2"]))
            for row in states
        ]
    )
    lattice = translation_lattice_factor_separable(
        qx, qy, qz, translations, super_metadata["supercell"]
    ).reshape(profile["nq"], profile["nphi"])

    acfo_setup_start = time.perf_counter()
    prepared = PreparedExactCoordinateHarmonicPlan(
        unit_coords,
        q_perp,
        q_z_rows,
        phi,
        element_indices=unit_e,
        form_factors=form_factors,
        harmonic_margin=int(acfo_contract["harmonic_margin"]),
        prepare_direct_basis=False,
        coefficient_backend=str(acfo_contract["coefficient_backend"]),
    )
    acfo_setup_s = time.perf_counter() - acfo_setup_start
    validation_plan = PreparedExactCoordinateHarmonicPlan(
        unit_coords,
        q_perp,
        q_z_rows,
        phi,
        element_indices=unit_e,
        form_factors=form_factors,
        harmonic_margin=int(acfo_contract["nested_validation_margin"]),
        prepare_direct_basis=False,
        coefficient_backend=str(acfo_contract["coefficient_backend"]),
    )
    if not prepared.fft_supported or not validation_plan.fft_supported:
        raise RuntimeError("ACFO cutoff reaches Nyquist under the frozen contract")

    def acfo_library(local_plan: PreparedExactCoordinateHarmonicPlan) -> np.ndarray:
        bases = []
        for weights in basis_weights:
            value, returned_cutoffs = local_plan.execute(
                atom_weights=weights,
                synthesis_backend=str(acfo_contract["synthesis_backend"]),
            )
            if not np.array_equal(returned_cutoffs, local_plan.cutoffs):
                raise RuntimeError("ACFO returned cutoffs differ from its prepared plan")
            bases.append(value * lattice)
        return synthesize_affine_candidate_library(
            np.stack(bases), occupancies, b_factors
        )

    acfo_value = acfo_library(prepared)
    validation_margin_value = acfo_library(validation_plan)

    # Generate the frozen off-grid observation with the full repeated supercell.
    truth_unit_weights = np.ones(unit_coords.shape[0], dtype=np.complex128)
    truth_unit_weights[occupancy_mask] = float(
        problem["truth"]["subdomain_occupancy"]
    )
    truth_super_weights = np.tile(truth_unit_weights, translations.shape[0])
    observation_plans, observation_masks = build_finufft_plans(
        super_coords,
        super_e,
        qx,
        qy,
        qz,
        n_elements=len(element_order),
        eps=float(finufft_contract["observation_epsilon"]),
        threads=int(finufft_contract["threads"]),
        n_trans=1,
    )
    truth_flat = execute_finufft_batched_plans(
        observation_plans,
        observation_masks,
        truth_super_weights[None, :],
        form_factors,
        target_q_indices,
    )[0]
    truth_amplitude = truth_flat.reshape(profile["nq"], profile["nphi"])
    truth_amplitude *= debye_waller_amplitude(
        q_report, float(problem["truth"]["b_iso_angstrom2"])
    )[:, None]
    observed_intensity, noise = add_intensity_noise(
        np.square(np.abs(truth_amplitude)),
        relative_l2=float(problem["intensity_noise_relative_l2"]),
        seed=int(problem["noise_seed"]),
    )
    del observation_plans, observation_masks, truth_flat, truth_amplitude
    gc.collect()

    def fused_library(plan: Any) -> np.ndarray:
        flat = execute_finufft_fused_element_plan(
            plan, unit_e, basis_weights, form_factors, target_q_indices
        )
        bases = flat.reshape(2, profile["nq"], profile["nphi"])
        bases *= lattice[None, ...]
        return synthesize_affine_candidate_library(bases, occupancies, b_factors)

    # Accuracy reference and epsilon frontier intentionally contain no timers.
    reference_plan = build_finufft_fused_element_plan(
        unit_coords,
        qx,
        qy,
        qz,
        n_elements=len(element_order),
        n_source_bases=2,
        eps=float(finufft_contract["accuracy_reference_epsilon"]),
        threads=int(finufft_contract["threads"]),
    )
    reference_value = fused_library(reference_plan)
    del reference_plan
    frontier_rows: list[dict[str, Any]] = []
    for index, eps in enumerate(eps_candidates):
        plan = build_finufft_fused_element_plan(
            unit_coords,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            n_source_bases=2,
            eps=eps,
            threads=int(finufft_contract["threads"]),
        )
        candidate_value = fused_library(plan)
        metrics = library_accuracy_metrics(
            acfo_value,
            candidate_value,
            reference_value,
            observed_intensity,
            top_k=int(problem["top_k"]),
        )
        gates = accuracy_gates(metrics, selection_contract["gates"])
        row = {
            "candidate_index": int(index),
            "eps": float(eps),
            "metrics": metrics,
            "gates": gates,
            "accuracy_pass": bool(all(gates.values())),
        }
        frontier_rows.append(row)
        print(
            f"epsilon {eps:.1e}: pass={row['accuracy_pass']} "
            f"ACFO/candidate I-L2={metrics['maximum_acfo_vs_candidate_intensity_relative_l2']:.6g} "
            f"candidate/reference I-L2={metrics['maximum_candidate_vs_reference_intensity_relative_l2']:.6g}",
            flush=True,
        )
        del plan, candidate_value
        gc.collect()

    selected_row = select_loosest_passing_epsilon(frontier_rows)
    selected_eps = None if selected_row is None else float(selected_row["eps"])

    acfo_reference_metrics = library_accuracy_metrics(
        acfo_value,
        reference_value,
        reference_value,
        observed_intensity,
        top_k=int(problem["top_k"]),
    )
    nested_complex = float(
        max(
            relative_l2(acfo_value[index], validation_margin_value[index])
            for index in range(len(states))
        )
    )
    nested_intensity = float(
        max(
            relative_l2(
                np.square(np.abs(acfo_value[index])),
                np.square(np.abs(validation_margin_value[index])),
            )
            for index in range(len(states))
        )
    )
    margin = int(acfo_contract["harmonic_margin"])
    kernel_cutoffs = prepared.cutoffs - margin
    harmonic_accuracy_gates = {
        "cutoffs_follow_frozen_qr_rule": bool(
            np.array_equal(
                kernel_cutoffs,
                np.ceil(np.abs(q_perp) * float(prepared.radius.max())).astype(np.int64),
            )
        ),
        "cutoffs_non_decreasing_with_qr": bool(np.all(np.diff(kernel_cutoffs) >= 0)),
        "selected_cutoff_below_nyquist": bool(
            int(prepared.max_cutoff) < profile["nphi"] // 2
        ),
        "nested_cutoff_below_nyquist": bool(
            int(validation_plan.max_cutoff) < profile["nphi"] // 2
        ),
        "acfo_vs_reference_intensity": bool(
            acfo_reference_metrics[
                "maximum_acfo_vs_reference_intensity_relative_l2"
            ]
            <= float(
                selection_contract["gates"]
                ["maximum_acfo_vs_reference_intensity_relative_l2"]
            )
        ),
        "nested_margin_intensity": bool(
            nested_intensity
            <= float(
                selection_contract["gates"]
                ["maximum_acfo_vs_reference_intensity_relative_l2"]
            )
        ),
    }

    final_timing = None
    selected_plan_setup_s = None
    selected_recheck = None
    if selected_eps is not None:
        setup_start = time.perf_counter()
        selected_plan = build_finufft_fused_element_plan(
            unit_coords,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            n_source_bases=2,
            eps=selected_eps,
            threads=int(finufft_contract["threads"]),
        )
        selected_plan_setup_s = time.perf_counter() - setup_start

        def acfo_execute() -> np.ndarray:
            return acfo_library(prepared)

        def selected_execute() -> np.ndarray:
            return fused_library(selected_plan)

        selected_value = selected_execute()
        selected_recheck = library_accuracy_metrics(
            acfo_value,
            selected_value,
            reference_value,
            observed_intensity,
            top_k=int(problem["top_k"]),
        )
        final_timing = measure_paired_blocks(
            acfo_execute,
            selected_execute,
            warmups=profile["warmups"],
            samples=profile["samples"],
            bootstrap_samples=profile["bootstrap_samples"],
            bootstrap_seed=int(protocol["final_timing"]["bootstrap_seed"]),
        )
        final_timing.update(
            {
                "finufft_plan_type": 3,
                "finufft_n_trans": 8,
                "selected_eps": selected_eps,
                "finufft_plan_setup_s_excluded": float(selected_plan_setup_s),
                "acfo_geometry_setup_s_excluded": float(acfo_setup_s),
                "selection_and_accuracy_excluded": True,
                "scoring_and_observation_generation_excluded": True,
            }
        )

    payload: dict[str, Any] = {
        "schema": "waxs-affine-library-eps-frontier-result-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if selected_eps is not None else "NO_ACCURACY_PASS",
        "mode": mode,
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
            "schema": protocol["schema"],
            "prospective_status": protocol["status"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "finufft_threads": int(finufft_contract["threads"]),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "problem": {
            "candidate_count": len(states),
            "q_range_inv_angstrom": [float(q_report[0]), float(q_report[-1])],
            "nq": profile["nq"],
            "nphi": profile["nphi"],
            "target_count": int(profile["nq"] * profile["nphi"]),
            "unit_atom_count": int(unit_coords.shape[0]),
            "supercell_atom_count": int(super_coords.shape[0]),
            "cell_count": int(translations.shape[0]),
            "element_channel_count": len(element_order),
            "source_basis_count": 2,
            "unit_structure_sha256": sha256(unit_path),
            "supercell_structure_sha256": sha256(supercell_path),
            "unit_structure_id": unit_metadata.get("structure_id"),
            "repetition_residual_nm": float(repetition_residual),
            "observed_intensity_sha256": array_sha256(observed_intensity),
            "noise": noise,
        },
        "representation_fairness": {
            "source_geometry": "8008 arbitrary atomic coordinates",
            "target_geometry": "nonuniform curved Ewald/cake targets",
            "lowest_applicable_nufft_type": 3,
            "same_unit_cell": True,
            "same_exact_finite_lattice_factor": True,
            "same_two_occupancy_source_bases": True,
            "same_four_element_channels": True,
            "b_factor_scaling_outside_fourier_kernel_for_both": True,
            "eligible_finufft_variant": "one fused n_trans=8 Type-3 plan",
        },
        "epsilon_selection": {
            "candidate_order": eps_candidates,
            "selection_rule": selection_contract["rule"],
            "timing_used": False,
            "timing_fields_recorded_per_candidate": False,
            "accuracy_reference_epsilon": float(
                finufft_contract["accuracy_reference_epsilon"]
            ),
            "gates": selection_contract["gates"],
            "rows": frontier_rows,
            "selected_eps": selected_eps,
            "selected_candidate_index": (
                None if selected_row is None else int(selected_row["candidate_index"])
            ),
            "accuracy_pass": selected_eps is not None,
        },
        "acfo_harmonic_accuracy": {
            "cutoff_rule": acfo_contract["harmonic_cutoff_rule"],
            "r_max_nm": float(prepared.radius.max()),
            "q_perp_r_max": (np.abs(q_perp) * float(prepared.radius.max())).tolist(),
            "kernel_cutoff": kernel_cutoffs.astype(int).tolist(),
            "harmonic_margin": margin,
            "compute_cutoff": prepared.cutoffs.astype(int).tolist(),
            "nested_validation_margin": int(acfo_contract["nested_validation_margin"]),
            "nested_compute_cutoff": validation_plan.cutoffs.astype(int).tolist(),
            "angular_nyquist_index": profile["nphi"] // 2,
            "miller_downward_recurrence_extra_order": int(
                acfo_contract["miller_downward_recurrence_extra_order"]
            ),
            "coefficient_backend": prepared.coefficient_backend,
            "synthesis_backend": acfo_contract["synthesis_backend"],
            "maximum_acfo_vs_reference_complex_relative_l2": float(
                acfo_reference_metrics[
                    "maximum_acfo_vs_reference_complex_relative_l2"
                ]
            ),
            "maximum_acfo_vs_reference_intensity_relative_l2": float(
                acfo_reference_metrics[
                    "maximum_acfo_vs_reference_intensity_relative_l2"
                ]
            ),
            "maximum_margin48_vs_margin64_complex_relative_l2": nested_complex,
            "maximum_margin48_vs_margin64_intensity_relative_l2": nested_intensity,
            "gates": harmonic_accuracy_gates,
            "accuracy_pass": all(harmonic_accuracy_gates.values()),
        },
        "selected_accuracy_recheck": selected_recheck,
        "final_timing": final_timing,
        "decision": {
            "epsilon_accuracy_selected_before_timing": True,
            "selected_fused_n_trans_8_only": True,
            "acfo_harmonic_accuracy_pass": all(harmonic_accuracy_gates.values()),
            "external_contract_complete": bool(
                mode == "full"
                and selected_eps is not None
                and final_timing is not None
                and final_timing["warmup_count"] == 10
                and final_timing["samples_per_arm"] == 30
            ),
            "speed_sign_is_integrity_gate": False,
        },
        "historical_receipt": {
            "path": "reports/acfo_p0_p1_external_return_v6_20260817/acfo_p0_p1_external_return/evidence/receipts/waxs_structured_library_closure.json",
            "role": "fixed-eps=1e-6 history; not overwritten or silently upgraded",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output, payload)
    csv_rows = []
    for row in frontier_rows:
        metrics = row["metrics"]
        ranking = metrics["acfo_vs_candidate_ranking"]
        csv_rows.append(
            {
                "candidate_index": row["candidate_index"],
                "eps": row["eps"],
                "accuracy_pass": row["accuracy_pass"],
                "maximum_acfo_vs_candidate_intensity_relative_l2": metrics[
                    "maximum_acfo_vs_candidate_intensity_relative_l2"
                ],
                "maximum_candidate_vs_reference_intensity_relative_l2": metrics[
                    "maximum_candidate_vs_reference_intensity_relative_l2"
                ],
                "maximum_acfo_vs_reference_intensity_relative_l2": metrics[
                    "maximum_acfo_vs_reference_intensity_relative_l2"
                ],
                "same_top1": ranking["same_top1"],
                "top_k_jaccard": ranking["top_k_overlap"]["jaccard"],
                "spearman": ranking["spearman"],
            }
        )
    write_csv(output.with_suffix(".csv"), csv_rows)
    speed = None if final_timing is None else final_timing["baseline_over_acfo_speedup"]
    md_lines = [
        "# WAXS affine-library epsilon-frontier closure",
        "",
        f"- mode: `{mode}`",
        f"- selected epsilon: `{selected_eps}`",
        f"- accuracy-first selection passed: `{selected_eps is not None}`",
        f"- ACFO harmonic audit passed: `{all(harmonic_accuracy_gates.values())}`",
        f"- ACFO/reference maximum intensity rel-L2: `{acfo_reference_metrics['maximum_acfo_vs_reference_intensity_relative_l2']:.6g}`",
        f"- margin 48/64 maximum intensity rel-L2: `{nested_intensity:.6g}`",
        "- the epsilon frontier records no timing values",
    ]
    if speed is not None:
        md_lines.extend(
            [
                f"- final warmups/samples per arm: `{profile['warmups']} / {profile['samples']}`",
                f"- selected fused Type-3 n_trans: `8`",
                f"- FINUFFT/ACFO median ratio: `{speed['point']:.6f}`",
                f"- paired-bootstrap 95% interval: `[{speed['lower_95']:.6f}, {speed['upper_95']:.6f}]`",
            ]
        )
    output.with_suffix(".md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accuracy-first epsilon frontier for the structured WAXS library."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results" / "waxs_100_state_eps_frontier.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = run(protocol_path, mode=args.mode, output=output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "selected_eps": payload["epsilon_selection"]["selected_eps"],
                "acfo_harmonic_accuracy_pass": payload["acfo_harmonic_accuracy"][
                    "accuracy_pass"
                ],
                "external_contract_complete": payload["decision"][
                    "external_contract_complete"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
