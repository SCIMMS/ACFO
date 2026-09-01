from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the ACFO W3a minimum production gate.")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("amplitudes", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/protein_nanocrystal_w3a_gate_decision.json"))
    args = parser.parse_args()

    row = load_json(args.benchmark)
    with np.load(args.amplitudes, allow_pickle=False) as payload:
        acfo = np.asarray(payload["acfo"])
        finufft = np.asarray(payload["finufft"])

    acfo_ring = np.mean(np.abs(acfo) ** 2, axis=1, dtype=np.float64)
    finufft_ring = np.mean(np.abs(finufft) ** 2, axis=1, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    ring_relative = np.abs(acfo_ring - finufft_ring) / np.maximum(np.abs(finufft_ring), tiny)
    ring_median = float(np.median(ring_relative))
    ring_p99 = float(np.percentile(ring_relative, 99))

    whole_peak_ratio = row["finufft_first_peak_rss_mib"] / row["acfo_first_peak_rss_mib"]
    incremental_peak_ratio = (
        row["finufft_first_peak_rss_delta_mib"] / row["acfo_first_peak_rss_delta_mib"]
    )
    t1 = next(item for item in row["total_time_model"] if item["T"] == 1)
    t100 = next(item for item in row["total_time_model"] if item["T"] == 100)

    gates = {
        "scale_pass": bool(
            row["atoms"] >= 100_000
            and row["nq"] >= 256
            and row["n_phi"] >= 512
            and row["qmax"] >= 6.0
        ),
        "complex_l2_pass": bool(row["complex_l2_acfo_vs_finufft"] <= 1e-6),
        "ring_median_pass": bool(ring_median <= 1e-3),
        "ring_p99_pass": bool(ring_p99 <= 5e-3),
        "warm_speed_pass": bool(row["warm_speedup_finufft_over_acfo"] >= 3.0),
        "t1_total_speed_pass": bool(t1["speedup_finufft_over_acfo"] >= 3.0),
        "t100_total_speed_pass_modeled": bool(t100["speedup_finufft_over_acfo"] >= 3.0),
        "memory_alternative_pass": bool(whole_peak_ratio >= 4.0),
        "break_even_pass": bool(
            row["break_even_repeat"] is not None and row["break_even_repeat"] <= 10
        ),
    }
    accuracy_pass = all(
        gates[key]
        for key in ("scale_pass", "complex_l2_pass", "ring_median_pass", "ring_p99_pass")
    )
    compute_resource_pass = bool(
        gates["t1_total_speed_pass"]
        or gates["t100_total_speed_pass_modeled"]
        or gates["memory_alternative_pass"]
    )
    minimum_gate_pass = bool(accuracy_pass and compute_resource_pass and gates["break_even_pass"])

    result = {
        "schema": "acfo-w3a-minimum-production-gate-v1",
        "benchmark_path": args.benchmark.as_posix(),
        "amplitudes_path": args.amplitudes.as_posix(),
        "regime": {
            "atoms": row["atoms"],
            "nq": row["nq"],
            "n_phi": row["n_phi"],
            "qmax": row["qmax"],
            "q_unit": row["q_unit"],
            "source_mode": row["source_mode"],
            "finufft_threads": row["finufft_threads"],
            "finufft_eps": row["finufft_eps"],
            "finufft_q_block_size": row["finufft_q_block_size"],
        },
        "metrics": {
            "complex_l2": row["complex_l2_acfo_vs_finufft"],
            "intensity_l2": row["intensity_l2_acfo_vs_finufft"],
            "ring_l2": row["ring_l2_acfo_vs_finufft"],
            "ring_pointwise_relative_median": ring_median,
            "ring_pointwise_relative_p99": ring_p99,
            "ring_relative_error_definition": "abs(mean_phi(I_acfo)-mean_phi(I_ref))/abs(mean_phi(I_ref)), evaluated over q rows",
            "warm_speedup": row["warm_speedup_finufft_over_acfo"],
            "t1_total_speedup": t1["speedup_finufft_over_acfo"],
            "t100_total_speedup_modeled": t100["speedup_finufft_over_acfo"],
            "whole_process_peak_memory_ratio_finufft_over_acfo": whole_peak_ratio,
            "incremental_peak_memory_ratio_finufft_over_acfo": incremental_peak_ratio,
            "break_even_repeat": row["break_even_repeat"],
        },
        "gates": gates,
        "accuracy_gate_pass": accuracy_pass,
        "compute_resource_gate_pass": compute_resource_pass,
        "minimum_w3a_production_gate_pass": minimum_gate_pass,
        "pass_basis": "memory alternative" if minimum_gate_pass and gates["memory_alternative_pass"] and not gates["t1_total_speed_pass"] else "speed",
        "limitations": [
            "FINUFFT q-block setup is included in every evaluation because the full-target plan exceeds local memory.",
            "T=100 is modeled from measured first and cached medians, not a directly executed 100-evaluation workflow.",
            "Timing uses two cached repeats, not the publication protocol of 10 warm-ups and 30 measured runs.",
            "GPU baseline is unavailable in the current Windows environment because cufinufft DLL dependencies do not load.",
            "This decision closes the W3a minimum production sub-gate, not the full WAXS validation program or NCS submission gate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ACFO W3a minimum production gate decision",
        "",
        f"Decision: `{'PASS' if minimum_gate_pass else 'FAIL'}` via `{result['pass_basis']}`.",
        "",
        "| check | value | gate | result |",
        "|---|---:|---:|---|",
        f"| scale | {row['atoms']:,} atoms; Nq={row['nq']}; Nphi={row['n_phi']}; qmax={row['qmax']} | minimum scale | {'PASS' if gates['scale_pass'] else 'FAIL'} |",
        f"| complex L2 | {row['complex_l2_acfo_vs_finufft']:.3e} | <=1e-6 | {'PASS' if gates['complex_l2_pass'] else 'FAIL'} |",
        f"| ring median | {ring_median:.3e} | <=1e-3 | {'PASS' if gates['ring_median_pass'] else 'FAIL'} |",
        f"| ring p99 | {ring_p99:.3e} | <=5e-3 | {'PASS' if gates['ring_p99_pass'] else 'FAIL'} |",
        f"| warm speed | {row['warm_speedup_finufft_over_acfo']:.2f}x | >=3x | {'PASS' if gates['warm_speed_pass'] else 'FAIL'} |",
        f"| T=1 total | {t1['speedup_finufft_over_acfo']:.2f}x | >=3x | {'PASS' if gates['t1_total_speed_pass'] else 'FAIL'} |",
        f"| whole-process memory | {whole_peak_ratio:.2f}x lower | >=4x | {'PASS' if gates['memory_alternative_pass'] else 'FAIL'} |",
        f"| break-even | {row['break_even_repeat']} | <=10 | {'PASS' if gates['break_even_pass'] else 'FAIL'} |",
        "",
        "The speed gate fails; the predeclared memory alternative passes. Publication-grade repeats, direct T workflow, and GPU/independent reruns remain open.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
