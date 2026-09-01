from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
INPUTS = [
    RESULTS / "protein_lattice_finufft_512_3x3.json",
    RESULTS / "protein_lattice_finufft_512_5x5.json",
]
OUTPUT = RESULTS / "protein_lattice_finufft_512.json"
SUMMARY = RESULTS / "protein_lattice_finufft_512.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_markdown(result: dict) -> None:
    lines = [
        "# Perfect protein crystal crossover vs FINUFFT, Nq=512",
        "",
        f"- unit atoms: `{result['unit_atom_count']:,}`",
        f"- targets: `{result['target_count']:,}`",
        f"- FINUFFT eps / threads: `{result['finufft_eps']:.0e} / {result['finufft_threads']}`",
        "",
        "| case | atoms | factorized first/hot s | FINUFFT first/hot s | factorized speedup first/hot | FINUFFT execute peak RSS delta | complex L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['label']} | {row['atom_count']:,} "
            f"| {row['factorized']['first_total_seconds']:.3f}/{row['factorized']['hot_seconds']['median']:.3f} "
            f"| {row['finufft']['first_total_seconds']:.3f}/{row['finufft']['hot_seconds']['median']:.3f} "
            f"| {row['speedup']['first_total_finufft_over_factorized']:.3f}x/{row['speedup']['hot_finufft_over_factorized']:.3f}x "
            f"| {row['finufft']['execute_peak_rss_delta_mib']:.1f} MiB "
            f"| {row['cross_error']['complex_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- crossover detected: **{'PASS' if result['regime_crossover_detected'] else 'FAIL'}**",
            f"- >=1M structured-regime performance gate: **{'PASS' if result['comparative_performance_pass'] else 'FAIL'}**",
            "- The 216k case favors FINUFFT; the 1.001M exact repeated crystal favors the factorized path.",
            "- This is a standard crystallographic specialization and a narrow regime claim, not a general ACFO or novelty claim.",
            "- Each hot timing is one measured repeat; independent-machine and repeated AB/BA timing remain pending.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in INPUTS]
    reference = payloads[0]
    matching_keys = (
        "schema",
        "unit_structure",
        "unit_atom_count",
        "q_inv_angstrom",
        "nq",
        "target_nphi",
        "target_count",
        "maximum_harmonic",
        "finufft_eps",
        "finufft_threads",
    )
    for payload in payloads[1:]:
        for key in matching_keys:
            if payload[key] != reference[key]:
                raise RuntimeError(f"split artifacts disagree on {key}")
    if not all(payload.get("validity_pass") for payload in payloads):
        raise RuntimeError("one or more split artifacts failed validity")
    rows = sorted(
        [row for payload in payloads for row in payload["rows"]],
        key=lambda row: row["atom_count"],
    )
    if len(rows) != 2:
        raise RuntimeError("expected exactly two crossover rows")
    small, large = rows
    small_first = small["speedup"]["first_total_finufft_over_factorized"]
    small_hot = small["speedup"]["hot_finufft_over_factorized"]
    large_first = large["speedup"]["first_total_finufft_over_factorized"]
    large_hot = large["speedup"]["hot_finufft_over_factorized"]
    regime_crossover = (
        small_first < 1.0
        and small_hot < 1.0
        and large_first > 1.0
        and large_hot > 1.0
    )
    performance_pass = (
        large["atom_count"] >= 1_000_000
        and large_first >= 3.0
        and large_hot >= 3.0
    )
    gates = {
        "split_artifacts_valid": True,
        "all_cross_complex_l2_le_2e_6": all(
            row["cross_error"]["complex_l2"] <= 2e-6 for row in rows
        ),
        "all_cross_intensity_l2_le_5e_6": all(
            row["cross_error"]["intensity_l2"] <= 5e-6 for row in rows
        ),
        "regime_crossover_detected": regime_crossover,
        "million_atom_first_speedup_ge_3": large_first >= 3.0,
        "million_atom_hot_speedup_ge_3": large_hot >= 3.0,
    }
    result = {
        "schema": "protein-lattice-finufft-512-crossover-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for path in INPUTS
        ],
        **{key: reference[key] for key in matching_keys if key != "schema"},
        "rows": rows,
        "gates": gates,
        "validity_pass": all(gates.values()),
        "regime_crossover_detected": regime_crossover,
        "comparative_performance_pass": performance_pass,
        "passed": all(gates.values()) and performance_pass,
        "decision": (
            "Keep only a narrow million-atom exact repeated-crystal performance claim; the 216k case remains FINUFFT-favorable."
        ),
        "claim_boundary": [
            "The finite lattice factor is the standard exact crystallographic specialization and is not an ACFO novelty claim.",
            "The supported performance regime is the tested 1.001M-atom exact repeated crystal, not arbitrary crystals or dense disorder.",
            "Direct NDFT correctness is established by the separate q=3/subset control; Nq=512 is an optimized-method cross-check.",
            "FINUFFT eps=1e-6 is a practical timing baseline, not a converged correctness oracle.",
            "Each Nq=512 hot timing has one measured repeat on the local machine; independent AB/BA repetition remains required before publication.",
        ],
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
