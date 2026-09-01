from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_median(rows: list[dict], order: str) -> float:
    values = [row["paired_speedup"] for row in rows if row["order"] == order]
    return float(np.median(values))


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / (0.5 * (a + b))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an independent-machine prepared fused 1M WAXS AB/BA receipt."
    )
    parser.add_argument("external_result", type=Path)
    parser.add_argument("external_environment", type=Path)
    parser.add_argument(
        "--local-result",
        type=Path,
        default=Path("benchmark_results/protein_lattice_prepared_finufft_512_abba.json"),
    )
    parser.add_argument(
        "--local-environment",
        type=Path,
        default=Path("benchmark_results/local_prepared_waxs_machine_environment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/external_prepared_waxs_abba_validation.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/external_prepared_waxs_abba_validation.md"),
    )
    args = parser.parse_args()
    paths = {
        "external_result": ROOT / args.external_result,
        "external_environment": ROOT / args.external_environment,
        "local_result": ROOT / args.local_result,
        "local_environment": ROOT / args.local_environment,
    }
    external = load(paths["external_result"])
    external_env = load(paths["external_environment"])
    local = load(paths["local_result"])
    local_env = load(paths["local_environment"])
    core_keys = (
        "unit",
        "supercell",
        "nq",
        "q_min_inv_angstrom",
        "q_max_inv_angstrom",
        "wavelength_nm",
        "target_nphi",
        "harmonic_margin",
        "finufft_eps",
        "finufft_threads",
        "warmup_pairs_requested",
        "measured_pairs_requested",
        "factorized_backend",
        "lattice_backend",
        "coefficient_backend",
        "synthesis_backend",
    )
    contract_match = all(
        external.get("contract", {}).get(key) == local["contract"].get(key)
        for key in core_keys
    )
    rows = external.get("measured_pairs", [])
    ab_count = sum(row.get("order") == "AB" for row in rows)
    ba_count = sum(row.get("order") == "BA" for row in rows)
    if ab_count and ba_count:
        ab_median = order_median(rows, "AB")
        ba_median = order_median(rows, "BA")
        order_gap = relative_gap(ab_median, ba_median)
    else:
        ab_median = None
        ba_median = None
        order_gap = None
    external_summary = external.get("measured_summary", {})
    speedup_summary = external_summary.get("paired_speedup", {})
    local_speedup = local["measured_summary"]["paired_speedup"]
    source_match = all(
        external_env.get("source_sha256", {}).get(key)
        == local_env.get("source_sha256", {}).get(key)
        for key in ("benchmark_driver", "cpp_solvers", "exact_harmonic")
    )
    fingerprints_differ = (
        external_env.get("machine_fingerprint_sha256")
        != local_env.get("machine_fingerprint_sha256")
    )
    gates = {
        "external_result_schema": external.get("schema")
        == "protein-lattice-prepared-finufft-512-abba-v1",
        "external_environment_schema": external_env.get("schema")
        == "prepared-waxs-machine-environment-v1",
        "external_status_complete": external.get("status") == "complete",
        "external_receipt_pass": bool(external.get("passed")),
        "core_contract_matches_local": contract_match,
        "source_hashes_match_local": source_match,
        "machine_fingerprint_differs": fingerprints_differ,
        "warmup_10_measured_30": external.get("warmup_pairs_completed") == 10
        and external.get("measured_pairs_completed") == 30,
        "balanced_15_ab_15_ba": ab_count == 15 and ba_count == 15,
        "order_speedup_median_gap_le_10pct": order_gap is not None
        and order_gap <= 0.10,
        "complex_l2_le_2e_6": external.get("cross_error", {}).get(
            "complex_l2", float("inf")
        )
        <= 2e-6,
        "legacy_l2_le_1e_12": external.get("legacy_comparison", {}).get(
            "legacy_vs_prepared_complex_l2", float("inf")
        )
        <= 1e-12,
        "paired_median_ge_3": speedup_summary.get("median", 0.0) >= 3.0,
        "paired_p05_ge_3": speedup_summary.get("p05", 0.0) >= 3.0,
    }
    result = {
        "schema": "external-prepared-waxs-abba-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "local_machine_fingerprint": local_env["machine_fingerprint_sha256"],
        "external_machine_fingerprint": external_env.get(
            "machine_fingerprint_sha256"
        ),
        "local_reference": {
            "paired_speedup_median": local_speedup["median"],
            "paired_speedup_p05": local_speedup["p05"],
        },
        "external_measurement": {
            "factorized_median_seconds": external_summary.get(
                "factorized_seconds", {}
            ).get("median"),
            "finufft_median_seconds": external_summary.get(
                "finufft_seconds", {}
            ).get("median"),
            "paired_speedup_median": speedup_summary.get("median"),
            "paired_speedup_p05": speedup_summary.get("p05"),
            "AB_speedup_median": ab_median,
            "BA_speedup_median": ba_median,
            "order_relative_gap": order_gap,
            "cross_error": external.get("cross_error"),
        },
        "gates": gates,
        "independent_machine_replication_pass": all(gates.values()),
        "claim_boundary": [
            "A different machine fingerprint and matching source hashes are required in addition to numerical/timing gates.",
            "The result applies only to the matched 1.001M-atom exact repeated-crystal contract.",
            "Dense disordered sources and other hardware are outside this timing claim.",
        ],
    }
    result["passed"] = result["independent_machine_replication_pass"]
    output = ROOT / args.output
    summary = ROOT / args.summary_md
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    order_gap_text = "N/A" if order_gap is None else f"{100 * order_gap:.3f}%"
    lines = [
        "# External prepared WAXS 10/30 validation",
        "",
        f"- independent-machine gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- machine fingerprint differs: `{fingerprints_differ}`",
        f"- source hashes match: `{source_match}`",
        f"- contract matches: `{contract_match}`",
        f"- external paired median / p05: `{speedup_summary.get('median')} / {speedup_summary.get('p05')}`",
        f"- AB/BA median gap: `{order_gap_text}`",
        "",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "gates": gates}, indent=2))
    print(f"wrote {output} and {summary}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
