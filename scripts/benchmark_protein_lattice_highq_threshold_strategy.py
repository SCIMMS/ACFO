from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_public_waxs_structures import load_structure  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402


DEFAULT_WINDOWS = (
    (0.05, 1.35),
    (1.70, 3.00),
    (3.35, 4.65),
    (5.00, 6.30),
    (6.70, 8.00),
)


def fft_friendly_even(value: int) -> int:
    target = max(4, int(value))
    if target % 2:
        target += 1
    while True:
        candidate = int(next_fast_len(target, real=True))
        if candidate % 2 == 0:
            return candidate
        target = candidate + 1


def physical_nphi(
    unit_coords: np.ndarray,
    q_max_inv_angstrom: float,
    *,
    wavelength_nm: float,
    harmonic_margin: int,
) -> tuple[int, int]:
    q_solver = np.asarray(
        [q_to_inv_nm(q_max_inv_angstrom, "inv_angstrom")], dtype=np.float64
    )
    q_perp, _ = ewald_ring(q_solver, wavelength_nm)
    radius = np.hypot(unit_coords[:, 0], unit_coords[:, 1])
    cutoff = int(math.ceil(abs(float(q_perp[0])) * float(radius.max())))
    cutoff += int(harmonic_margin)
    return fft_friendly_even(2 * cutoff + 2), cutoff


def parse_csv_ints(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out or any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def run_case(
    *,
    label: str,
    axis: str,
    q_min: float,
    q_max: float,
    nq: int,
    nphi: int,
    q_block: int,
    wall_threshold_s: float,
    memory_sample_interval_s: float,
    max_process_rss_fraction: float,
    min_available_memory_fraction: float,
    finufft_mode: str,
    resume: bool,
) -> dict:
    output = RESULTS / f"protein_lattice_highq_threshold_{label}.json"
    if resume and output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        contract = saved.get("contract", {})
        if (
            contract.get("q_min_inv_angstrom") == q_min
            and contract.get("q_max_inv_angstrom") == q_max
            and contract.get("nq") == nq
            and contract.get("nphi") == nphi
            and contract.get("finufft_q_block_size") == q_block
            and contract.get("coefficient_backend") == "fused_phase"
            and contract.get("lattice_backend") == "separable"
            and contract.get("finufft_mode") == finufft_mode
        ):
            print(f"reuse {output.name}", flush=True)
            return saved

    command = [
        sys.executable,
        "scripts/benchmark_protein_lattice_q_sampling_case.py",
        "--label",
        label,
        "--axis",
        axis,
        "--q-min",
        str(q_min),
        "--q-max",
        str(q_max),
        "--nq",
        str(nq),
        "--nphi",
        str(nphi),
        "--finufft-mode",
        finufft_mode,
        "--finufft-q-block-size",
        str(q_block),
        "--finufft-wall-threshold-s",
        str(wall_threshold_s),
        "--memory-sample-interval-s",
        str(memory_sample_interval_s),
        "--finufft-max-process-rss-fraction",
        str(max_process_rss_fraction),
        "--finufft-min-available-memory-fraction",
        str(min_available_memory_fraction),
        "--lattice-backend",
        "separable",
        "--coefficient-backend",
        "fused_phase",
        "--output",
        output.as_posix(),
    ]
    print(
        f"run {label}: q={q_min:.2f}-{q_max:.2f}, Nq={nq}, "
        f"Nphi={nphi}, q_block={q_block}, mode={finufft_mode}",
        flush=True,
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(120.0, wall_threshold_s + 120.0),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(
            json.dumps(
                {
                    "label": label,
                    "returncode": completed.returncode,
                    "elapsed_seconds": elapsed,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
                indent=2,
            )
        )
    print(completed.stdout[-1200:], flush=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["driver_subprocess"] = {
        "elapsed_seconds": elapsed,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def finufft_row(case: dict) -> dict:
    return case["timing_seconds"]["finufft"]


def peak_process_rss_fraction(case: dict) -> float | None:
    rows = finufft_row(case).get("block_rows", [])
    values = [
        row["peak_process_rss_fraction"]
        for row in rows
        if row.get("peak_process_rss_fraction") is not None
    ]
    return max(values) if values else None


def minimum_available_fraction(case: dict) -> float | None:
    rows = finufft_row(case).get("block_rows", [])
    values = [
        row["physical_memory_after_cleanup"]["available_fraction"]
        for row in rows
        if row.get("physical_memory_after_cleanup", {}).get("available_fraction")
        is not None
    ]
    return min(values) if values else None


def full_case_summary(case: dict) -> dict:
    finufft = finufft_row(case)
    factorized = case["timing_seconds"]["factorized"]
    wall = (
        None
        if finufft.get("mode") == "skip"
        else finufft.get("streamed_wall_excluding_shared")
    )
    return {
        "label": case["label"],
        "q_min_inv_angstrom": case["contract"]["q_min_inv_angstrom"],
        "q_max_inv_angstrom": case["contract"]["q_max_inv_angstrom"],
        "q_center_inv_angstrom": 0.5
        * (
            case["contract"]["q_min_inv_angstrom"]
            + case["contract"]["q_max_inv_angstrom"]
        ),
        "nq": case["contract"]["nq"],
        "nphi": case["contract"]["nphi"],
        "q_block_size": case["contract"].get("finufft_q_block_size"),
        "factorized_specific_setup_seconds": factorized["specific_setup_total"],
        "factorized_first_total_seconds": factorized[
            "first_total_excluding_shared"
        ],
        "factorized_hot_seconds": factorized["hot_execute"],
        "finufft_wall_seconds": wall,
        "finufft_completed_nq": finufft.get("completed_nq"),
        "finufft_completion_fraction": finufft.get("completion_fraction"),
        "finufft_censored": finufft.get("censored", False),
        "finufft_stop_reasons": finufft.get("stop_reasons", []),
        "measured_speedup_or_lower_bound": (
            None
            if wall is None
            else wall / factorized["first_total_excluding_shared"]
        ),
        "speedup_is_lower_bound": finufft.get("censored", False),
        "cross_complex_l2": (
            None if case.get("cross_error") is None else case["cross_error"]["complex_l2"]
        ),
        "peak_process_rss_fraction": peak_process_rss_fraction(case),
        "minimum_available_memory_fraction": minimum_available_fraction(case),
        "passed": case["passed"],
    }


def block_holdout_model(cases: list[dict], holdout_fraction: float) -> dict:
    observations = []
    for case in cases:
        for row in finufft_row(case).get("block_rows", []):
            observations.append(
                {
                    "q_center": 0.5
                    * (
                        row["q_min_inv_angstrom"]
                        + row["q_max_inv_angstrom"]
                    ),
                    "wall_seconds": row["block_wall_seconds"],
                    "target_count": row["target_count"],
                }
            )
    observations.sort(key=lambda row: row["q_center"])
    if len(observations) < 5:
        return {
            "available": False,
            "reason": "fewer than five measured high-q blocks",
            "passed": False,
        }
    n_holdout = max(1, int(math.ceil(len(observations) * holdout_fraction)))
    train = observations[:-n_holdout]
    holdout = observations[-n_holdout:]
    train_q = np.asarray([row["q_center"] for row in train], dtype=np.float64)
    train_times = np.asarray(
        [row["wall_seconds"] for row in train], dtype=np.float64
    )
    design = np.column_stack((np.ones(train_q.size), train_q))
    intercept, q_slope = np.linalg.lstsq(design, train_times, rcond=None)[0]
    train_prediction = intercept + q_slope * train_q
    holdout_q = np.asarray(
        [row["q_center"] for row in holdout], dtype=np.float64
    )
    holdout_prediction = intercept + q_slope * holdout_q
    actual_holdout = float(sum(row["wall_seconds"] for row in holdout))
    predicted_holdout = float(np.sum(holdout_prediction))
    relative_error = abs(predicted_holdout - actual_holdout) / actual_holdout
    absolute_residuals = np.abs(train_times - train_prediction)
    residual_p95 = float(np.quantile(absolute_residuals, 0.95))
    return {
        "available": True,
        "model": "linear completed block wall time versus q-center at fixed high-q window, Nphi, and q_block",
        "training_block_count": len(train),
        "holdout_block_count": len(holdout),
        "training_q_center_range": [train[0]["q_center"], train[-1]["q_center"]],
        "holdout_q_center_range": [holdout[0]["q_center"], holdout[-1]["q_center"]],
        "intercept_seconds": float(intercept),
        "q_center_slope_seconds_per_inv_angstrom": float(q_slope),
        "training_absolute_residual_p95_seconds": residual_p95,
        "actual_holdout_seconds": actual_holdout,
        "predicted_holdout_seconds": predicted_holdout,
        "holdout_relative_error": relative_error,
        "relative_error_gate": 0.15,
        "passed": relative_error <= 0.15,
    }


def predict_resolution_finufft_seconds(
    *,
    nq: int,
    q_min: float,
    q_max: float,
    q_block: int,
    model: dict,
) -> tuple[float, int]:
    q_values = np.linspace(q_min, q_max, nq, dtype=np.float64)
    q_centers = []
    for q_start in range(0, nq, q_block):
        q_stop = min(q_start + q_block, nq)
        q_centers.append(float(np.mean(q_values[q_start:q_stop])))
    predictions = (
        model["intercept_seconds"]
        + model["q_center_slope_seconds_per_inv_angstrom"]
        * np.asarray(q_centers)
    )
    predictions = np.maximum(predictions, 0.0)
    return float(np.sum(predictions)), len(q_centers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential high-q WAXS benchmark with explicit time/memory censoring."
    )
    parser.add_argument("--position-nq", type=int, default=16)
    parser.add_argument(
        "--resolution-nq", type=parse_csv_ints, default=parse_csv_ints("32,64,128,256,512")
    )
    parser.add_argument(
        "--q-block-candidates", type=parse_csv_ints, default=parse_csv_ints("1,2,4")
    )
    parser.add_argument("--wall-threshold-s", type=float, default=180.0)
    parser.add_argument("--calibration-threshold-s", type=float, default=90.0)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.02)
    parser.add_argument("--max-process-rss-fraction", type=float, default=0.75)
    parser.add_argument("--min-available-memory-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_lattice_highq_threshold_strategy.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/protein_lattice_highq_threshold_strategy.md"),
    )
    args = parser.parse_args()
    if args.position_nq < 2:
        raise ValueError("position-nq must be at least two")
    if args.wall_threshold_s <= 0 or args.calibration_threshold_s <= 0:
        raise ValueError("time thresholds must be positive")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("holdout-fraction must be in (0, 0.5)")

    suite_start = time.perf_counter()
    unit_coords, _, _ = load_structure(
        Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz")
    )

    window_contracts = []
    for q_min, q_max in DEFAULT_WINDOWS:
        nphi, cutoff = physical_nphi(
            unit_coords,
            q_max,
            wavelength_nm=args.wavelength_nm,
            harmonic_margin=args.harmonic_margin,
        )
        window_contracts.append(
            {
                "q_min": q_min,
                "q_max": q_max,
                "nphi": nphi,
                "maximum_harmonic_estimate": cutoff,
            }
        )

    high_window = window_contracts[-1]
    calibration_cases = []
    for q_block in args.q_block_candidates:
        case = run_case(
            label=f"qblock{q_block}_nq8_q6p70_8p00",
            axis="resolution",
            q_min=high_window["q_min"],
            q_max=high_window["q_max"],
            nq=8,
            nphi=high_window["nphi"],
            q_block=min(q_block, 8),
            wall_threshold_s=args.calibration_threshold_s,
            memory_sample_interval_s=args.memory_sample_interval_s,
            max_process_rss_fraction=args.max_process_rss_fraction,
            min_available_memory_fraction=args.min_available_memory_fraction,
            finufft_mode="chunked",
            resume=args.resume,
        )
        calibration_cases.append(case)

    safe_calibration = []
    for case in calibration_cases:
        finufft = finufft_row(case)
        peak_fraction = peak_process_rss_fraction(case)
        available_fraction = minimum_available_fraction(case)
        if (
            case["passed"]
            and not finufft["censored"]
            and (peak_fraction is None or peak_fraction < args.max_process_rss_fraction)
            and (
                available_fraction is None
                or available_fraction > args.min_available_memory_fraction
            )
        ):
            safe_calibration.append(case)
    if not safe_calibration:
        raise RuntimeError("no q-block candidate completed within the time/memory contract")
    selected_case = min(
        safe_calibration,
        key=lambda row: finufft_row(row)["streamed_wall_excluding_shared"],
    )
    selected_q_block = selected_case["contract"]["finufft_q_block_size"]
    print(f"selected q_block={selected_q_block}", flush=True)

    position_cases = []
    for index, window in enumerate(window_contracts):
        case = run_case(
            label=f"window{index+1}_nq{args.position_nq}",
            axis="range",
            q_min=window["q_min"],
            q_max=window["q_max"],
            nq=args.position_nq,
            nphi=window["nphi"],
            q_block=min(selected_q_block, args.position_nq),
            wall_threshold_s=args.wall_threshold_s,
            memory_sample_interval_s=args.memory_sample_interval_s,
            max_process_rss_fraction=args.max_process_rss_fraction,
            min_available_memory_fraction=args.min_available_memory_fraction,
            finufft_mode="chunked",
            resume=args.resume,
        )
        position_cases.append(case)

    resolution_cases = []
    censored_seen = False
    for nq in args.resolution_nq:
        finufft_mode = "skip" if censored_seen else "chunked"
        case = run_case(
            label=f"resolution_nq{nq}_q6p70_8p00",
            axis="resolution",
            q_min=high_window["q_min"],
            q_max=high_window["q_max"],
            nq=nq,
            nphi=high_window["nphi"],
            q_block=min(selected_q_block, nq),
            wall_threshold_s=args.wall_threshold_s,
            memory_sample_interval_s=args.memory_sample_interval_s,
            max_process_rss_fraction=args.max_process_rss_fraction,
            min_available_memory_fraction=args.min_available_memory_fraction,
            finufft_mode=finufft_mode,
            resume=args.resume,
        )
        resolution_cases.append(case)
        if finufft_mode == "chunked" and finufft_row(case)["censored"]:
            censored_seen = True

    measured_highq_cases = [position_cases[-1]] + [
        case
        for case in resolution_cases
        if finufft_row(case).get("mode") == "chunked_streaming"
    ]
    holdout = block_holdout_model(measured_highq_cases, args.holdout_fraction)

    resolution_rows = []
    for case in resolution_cases:
        row = full_case_summary(case)
        if holdout.get("passed") and (
            row["finufft_censored"] or row["finufft_wall_seconds"] is None
        ):
            predicted, block_count = predict_resolution_finufft_seconds(
                nq=row["nq"],
                q_min=high_window["q_min"],
                q_max=high_window["q_max"],
                q_block=selected_q_block,
                model=holdout,
            )
            interval_delta = holdout["training_absolute_residual_p95_seconds"] * block_count
            row["extrapolation"] = {
                "used": True,
                "predicted_full_finufft_seconds": predicted,
                "prediction_interval_seconds": [
                    max(0.0, predicted - interval_delta),
                    predicted + interval_delta,
                ],
                "predicted_speedup": predicted
                / row["factorized_first_total_seconds"],
            }
        elif row["finufft_wall_seconds"] is not None:
            row["extrapolation"] = {
                "used": False,
                "reason": "complete measured FINUFFT timing is available",
            }
        else:
            row["extrapolation"] = {
                "used": False,
                "reason": "holdout error gate did not pass",
            }
        resolution_rows.append(row)

    calibration_rows = [full_case_summary(case) for case in calibration_cases]
    position_rows = [full_case_summary(case) for case in position_cases]
    result = {
        "schema": "protein-lattice-highq-threshold-strategy-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "atom_count": position_cases[-1]["atom_count"],
            "unit_atom_count": position_cases[-1]["unit_atom_count"],
            "q_window_width_inv_angstrom": 1.30,
            "position_nq": args.position_nq,
            "resolution_nq": args.resolution_nq,
            "wall_threshold_seconds": args.wall_threshold_s,
            "calibration_threshold_seconds": args.calibration_threshold_s,
            "max_process_rss_fraction": args.max_process_rss_fraction,
            "min_available_memory_fraction": args.min_available_memory_fraction,
            "finufft_eps": 1e-6,
            "finufft_threads": 4,
            "coefficient_backend": "fused_phase",
            "lattice_backend": "separable",
            "nphi_rule": "minimum even FFT-friendly length satisfying estimated exact-coordinate harmonic cutoff",
        },
        "q_block_calibration": {
            "candidates": args.q_block_candidates,
            "selected_q_block": selected_q_block,
            "rows": calibration_rows,
        },
        "position_sweep": {
            "windows": window_contracts,
            "rows": position_rows,
        },
        "resolution_sweep": {
            "q_window": [high_window["q_min"], high_window["q_max"]],
            "rows": resolution_rows,
        },
        "extrapolation_validation": holdout,
        "gates": {
            "all_calibration_cases_pass": all(case["passed"] for case in calibration_cases),
            "safe_q_block_selected": bool(safe_calibration),
            "all_position_cases_complete": all(
                case["passed"] and not finufft_row(case)["censored"]
                for case in position_cases
            ),
            "all_measured_accuracy_l2_le_2e_6": all(
                case.get("cross_error") is None
                or case["cross_error"]["complex_l2"] <= 2e-6
                for case in calibration_cases + position_cases + resolution_cases
            ),
            "time_threshold_produces_censored_or_all_complete": (
                censored_seen
                or all(
                    finufft_row(case).get("mode") == "skip"
                    or not finufft_row(case)["censored"]
                    for case in resolution_cases
                )
            ),
            "extrapolation_holdout_relative_error_le_15pct": bool(
                holdout.get("passed")
            ),
        },
        "claim_boundary": [
            "Completed rows are local measured timings; censored rows report measured speedup lower bounds.",
            "Extrapolated FINUFFT totals are secondary planning estimates and are emitted only after a <=15% holdout gate.",
            "FINUFFT eps=1e-6 is a practical timing baseline; direct NDFT correctness remains separate.",
            "The repeated-crystal lattice factor is a crystallographic specialization, not a universal NUFFT replacement claim.",
        ],
        "elapsed_seconds": time.perf_counter() - suite_start,
    }
    result["passed"] = all(result["gates"].values())

    lines = [
        "# High-q threshold and censored FINUFFT strategy",
        "",
        f"- selected q block: `{selected_q_block}`",
        f"- time threshold: `{args.wall_threshold_s:.0f} s`",
        f"- holdout extrapolation gate: `{'PASS' if holdout.get('passed') else 'FAIL'}`",
        "",
        "## Equal-width q-window position sweep",
        "",
        "| q window (A^-1) | Nq/Nphi | ACFO first-total s | FINUFFT measured s | speedup | peak RSS / total | complex L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in position_rows:
        peak = row["peak_process_rss_fraction"]
        lines.append(
            f"| {row['q_min_inv_angstrom']:.2f}-{row['q_max_inv_angstrom']:.2f} "
            f"| {row['nq']}/{row['nphi']} | {row['factorized_first_total_seconds']:.3f} "
            f"| {row['finufft_wall_seconds']:.3f} | {row['measured_speedup_or_lower_bound']:.1f}x "
            f"| {'n/a' if peak is None else f'{100*peak:.1f}%'} "
            f"| {row['cross_complex_l2']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## High-q resolution sweep",
            "",
            "| Nq/Nphi | ACFO first-total s | FINUFFT measured s | status | measured speedup/lower bound | extrapolated full s |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in resolution_rows:
        measured = row["finufft_wall_seconds"]
        status = (
            "ACFO only"
            if measured is None
            else "censored" if row["finufft_censored"] else "complete"
        )
        projected = row["extrapolation"].get("predicted_full_finufft_seconds")
        measured_text = "-" if measured is None else f"{measured:.3f}"
        speedup = row["measured_speedup_or_lower_bound"]
        speedup_text = "-" if speedup is None else f"{speedup:.1f}x"
        projected_text = "-" if projected is None else f"{projected:.1f}"
        lines.append(
            f"| {row['nq']}/{row['nphi']} | {row['factorized_first_total_seconds']:.3f} "
            f"| {measured_text} | {status} "
            f"| {speedup_text} "
            f"| {projected_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Measured rows and timeout-derived lower bounds are primary evidence.",
            "- Extrapolated rows are secondary and must remain visually and verbally distinct.",
            "- Timeout or memory censoring is not converted into an exact FINUFFT runtime.",
            "",
            f"Overall gate: **{'PASS' if result['passed'] else 'FAIL'}**",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "gates": result["gates"]}, indent=2))
    print(f"wrote {args.output} and {args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
