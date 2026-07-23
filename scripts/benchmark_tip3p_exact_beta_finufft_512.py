from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_physical_scaling import timed_call  # noqa: E402
from benchmark_protein_nanocrystal_finufft_fair import build_finufft_plans  # noqa: E402
from prepare_openmm_water_waxs_inputs import (  # noqa: E402
    center_from_box,
    iter_dcd_frames,
    load_npz,
)
from validate_public_waxs_structures import build_form_factors  # noqa: E402
from waxs_cake import (  # noqa: E402
    encode_elements,
    exact_coordinate_harmonic_amplitude_factorized,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timing_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def write_markdown(result: dict, path: Path) -> None:
    finufft = result["finufft"]
    fused = result["cpp_fused"]
    lines = [
        "# TIP3P exact-beta vs FINUFFT at Nq=512",
        "",
        f"- atoms / targets: `{result['atom_count']:,} / {result['target_count']:,}`",
        f"- FINUFFT eps / threads: `{finufft['eps']:.0e} / {finufft['threads']}`",
        "",
        "| method | setup s | first/hot s | peak RSS delta MiB |",
        "|---|---:|---:|---:|",
        f"| cpp_fused exact-beta | 0 | {fused['seconds']:.3f} | {fused['memory']['peak_rss_delta_mib']:.1f} |",
        f"| FINUFFT reusable plan | {finufft['setup_seconds']:.3f} | {finufft['first_seconds']:.3f} / {finufft['hot_seconds']['median']:.3f} | {finufft['setup_memory']['peak_rss_delta_mib']:.1f} setup |",
        "",
        f"- fused vs FINUFFT complex/intensity L2: `{result['cross_error']['complex_l2']:.3e} / {result['cross_error']['intensity_l2']:.3e}`",
        f"- FINUFFT/fused hot-time ratio: `{result['finufft_hot_over_fused_ratio']:.3f}x`",
        f"- first-total FINUFFT/fused ratio: `{result['finufft_first_total_over_fused_ratio']:.3f}x`",
        f"- benchmark gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- comparative performance: **{'PASS' if result['comparative_performance_pass'] else 'FAIL'}**",
        "",
        "Direct NDFT remains the small-case correctness oracle. At Nq=512 the two optimized methods are cross-checked, not promoted to an independent oracle.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched 50k-atom Nq=512 exact-beta versus reusable FINUFFT comparison."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_final.npz"),
    )
    parser.add_argument(
        "--trajectory-dcd",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_trajectory.dcd"),
    )
    parser.add_argument("--nq", type=int, default=512)
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--finufft-hot-repeats", type=int, default=3)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_finufft_512.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_finufft_512.md"),
    )
    args = parser.parse_args()

    _, elements, _, source_metadata = load_npz(args.input_npz)
    element_indices, element_order = encode_elements(elements)
    frame_rows = list(iter_dcd_frames(args.trajectory_dcd, frame_indices={0}))
    if len(frame_rows) != 1:
        raise RuntimeError("failed to read trajectory frame 0")
    _, frame_coords, frame_box = frame_rows[0]
    coords = np.ascontiguousarray(
        frame_coords - center_from_box(frame_coords, frame_box), dtype=np.float64
    )
    atom_weights = np.ones(coords.shape[0], dtype=np.complex128)
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    ff_mapping = build_form_factors(elements, q_solver, "xray_f0")
    form_factors = normalize_form_factors(
        element_order, q_solver, ff_mapping
    ).astype(np.complex128, copy=False)
    phi = (np.arange(args.target_nphi) + 0.5) * (
        2.0 * np.pi / args.target_nphi
    )
    qx = np.ascontiguousarray(
        (q_perp[:, None] * np.cos(phi)[None, :]).ravel()
    )
    qy = np.ascontiguousarray(
        (q_perp[:, None] * np.sin(phi)[None, :]).ravel()
    )
    qz = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], (args.nq, args.target_nphi)).ravel()
    )
    target_q_indices = np.repeat(np.arange(args.nq), args.target_nphi)

    def fused_execute():
        return exact_coordinate_harmonic_amplitude_factorized(
            coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=element_indices,
            form_factors=form_factors,
            atom_weights=atom_weights,
            harmonic_margin=args.harmonic_margin,
        )

    (fused_output, cutoffs), fused_seconds, fused_memory = timed_call(
        fused_execute,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )

    (plans, masks), setup_seconds, setup_memory = timed_call(
        lambda: build_finufft_plans(
            coords,
            element_indices,
            qx,
            qy,
            qz,
            n_elements=len(element_order),
            eps=args.finufft_eps,
            threads=args.finufft_threads,
        ),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )

    def finufft_execute() -> np.ndarray:
        active = np.zeros(qx.size, dtype=np.complex128)
        for element_index, (plan, mask) in enumerate(zip(plans, masks)):
            values = plan.execute(np.ascontiguousarray(atom_weights[mask]))
            active += values * form_factors[element_index, target_q_indices]
        return active.reshape(args.nq, args.target_nphi)

    finufft_first, first_seconds, first_memory = timed_call(
        finufft_execute,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    hot_times = []
    finufft_hot = finufft_first
    for _ in range(args.finufft_hot_repeats):
        gc.collect()
        start = time.perf_counter()
        finufft_hot = finufft_execute()
        hot_times.append(time.perf_counter() - start)

    cross_error = {
        "complex_l2": relative_l2(finufft_hot, fused_output),
        "intensity_l2": relative_l2(
            intensity(finufft_hot), intensity(fused_output)
        ),
    }
    hot_summary = timing_summary(hot_times)
    gates = {
        "maximum_harmonic_below_target_nyquist": (
            int(np.max(cutoffs)) < args.target_nphi // 2
        ),
        "cross_complex_l2_le_2e_6": cross_error["complex_l2"] <= 2e-6,
        "cross_intensity_l2_le_5e_6": cross_error["intensity_l2"] <= 5e-6,
        "finite_positive_timings": all(
            np.isfinite(value) and value > 0
            for value in (
                fused_seconds,
                setup_seconds,
                first_seconds,
                hot_summary["median"],
            )
        ),
    }
    result = {
        "schema": "tip3p-exact-beta-finufft-512-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_npz": args.input_npz.as_posix(),
            "trajectory_dcd": args.trajectory_dcd.as_posix(),
            "frame_index": 0,
            "source_metadata": source_metadata,
        },
        "atom_count": int(coords.shape[0]),
        "q_inv_angstrom": [args.q_min, args.q_max],
        "nq": args.nq,
        "target_nphi": args.target_nphi,
        "target_count": int(qx.size),
        "maximum_harmonic": int(np.max(cutoffs)),
        "cpp_fused": {
            "seconds": fused_seconds,
            "memory": fused_memory,
        },
        "finufft": {
            "eps": args.finufft_eps,
            "threads": args.finufft_threads,
            "plan_count": len(plans),
            "setup_seconds": setup_seconds,
            "setup_memory": setup_memory,
            "first_seconds": first_seconds,
            "first_memory": first_memory,
            "hot_repeats": args.finufft_hot_repeats,
            "hot_seconds": hot_summary,
        },
        "cross_error": cross_error,
        "finufft_hot_over_fused_ratio": hot_summary["median"] / fused_seconds,
        "finufft_first_total_over_fused_ratio": (
            setup_seconds + first_seconds
        )
        / fused_seconds,
        "comparative_performance_pass": (
            fused_seconds <= hot_summary["median"]
        ),
        "performance_decision": (
            "FAIL: reusable FINUFFT is substantially faster for this dense exact-coordinate Nq=512 workload."
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            "Direct NDFT is the correctness oracle on small cases; this Nq=512 row is an optimized-method cross-check.",
            "FINUFFT eps=1e-6 is a practical timing baseline, not a converged numerical oracle.",
            "Timings and sampled RSS are local to this machine, build, thread policy, and run.",
            "The full polar grid is used rather than a detector rectangle so both methods produce identical targets.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.summary_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
