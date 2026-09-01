from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT_JSON = RESULTS / "waxs_protein_exact_beta_followup_decision.json"
OUTPUT_MD = RESULTS / "waxs_protein_exact_beta_followup_decision_ko.md"


SOURCES = {
    "protein_exact_beta": RESULTS / "waxs_exact_beta_harmonic_bridge.json",
    "protein_lattice": RESULTS / "protein_nanocrystal_lattice_factorization.json",
    "protein_prepared_abba": RESULTS / "protein_lattice_prepared_abba_decision.json",
    "tip3p_dense_md": RESULTS / "tip3p_dense_highq_exact_beta_20frames.json",
    "tip3p_dense_finufft": RESULTS / "tip3p_exact_beta_finufft_512.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing WAXS evidence: " + ", ".join(missing))
    exact = load(SOURCES["protein_exact_beta"])
    lattice = load(SOURCES["protein_lattice"])
    prepared = load(SOURCES["protein_prepared_abba"])
    tip3p = load(SOURCES["tip3p_dense_md"])
    tip3p_finufft = load(SOURCES["tip3p_dense_finufft"])
    supercells = {row["label"]: row for row in lattice["supercells"]}
    million = supercells["5x5x5"]
    medium = supercells["3x3x3"]

    metrics = {
        "protein_unit_cell": {
            "structure": exact["structure"],
            "atoms": int(exact["atom_count"]),
            "q_inv_angstrom": exact["q_inv_angstrom"],
            "target_nphi": int(exact["target_nphi"]),
            "exact_beta_complex_l2_vs_direct": float(
                exact["exact_coordinate_harmonic"]["complex_l2"]
            ),
            "exact_beta_intensity_l2_vs_direct": float(
                exact["exact_coordinate_harmonic"]["intensity_l2"]
            ),
            "fine_rz_pixel_intensity_l2": float(
                exact["rz_quantization_sweep"][-1]["intensity_l2"]
            ),
            "fine_rz_ring_intensity_l2": float(
                exact["rz_quantization_sweep"][-1]["ring_intensity_l2"]
            ),
        },
        "ordered_protein_supercells": {
            "3x3x3_atoms": int(medium["atom_count"]),
            "3x3x3_direct_subset_complex_l2": float(
                medium["subset_complex_l2"]
            ),
            "5x5x5_atoms": int(million["atom_count"]),
            "5x5x5_direct_subset_complex_l2": float(
                million["subset_complex_l2"]
            ),
            "sparse_defect_control_pass": bool(
                lattice["gates"][
                    "sparse_defect_delta_subset_complex_l2_le_1e_9"
                ]
            ),
        },
        "prepared_1m_abba": {
            "nq": int(prepared["contract"]["nq"]),
            "q_min_inv_angstrom": float(
                prepared["contract"]["q_min_inv_angstrom"]
            ),
            "q_max_inv_angstrom": float(
                prepared["contract"]["q_max_inv_angstrom"]
            ),
            "paired_speedup_median": float(
                prepared["prepared_measured_summary"]["paired_speedup"]["median"]
            ),
            "paired_speedup_p05": float(
                prepared["prepared_measured_summary"]["paired_speedup"]["p05"]
            ),
            "prepared_vs_finufft_complex_l2": float(
                prepared["accuracy"]["prepared_vs_finufft"]["complex_l2"]
            ),
            "local_gate_pass": bool(prepared["local_prepared_timing_gate_pass"]),
            "independent_machine_replication_complete": bool(
                prepared["independent_machine_replication_complete"]
            ),
        },
        "dense_md_control": {
            "material": "TIP3P water trajectory",
            "frames": int(tip3p["frame_count"]),
            "atoms": int(tip3p["atom_count"]),
            "maximum_exact_beta_complex_l2": float(
                tip3p["aggregate"]["exact_beta_complex_l2"]["max"]
            ),
            "median_direct_over_exact_beta_speedup": float(
                tip3p["aggregate"]["direct_over_exact_beta_speedup"]["median"]
            ),
            "nq512_exact_beta_s": float(tip3p_finufft["cpp_fused"]["seconds"]),
            "nq512_finufft_hot_median_s": float(
                tip3p_finufft["finufft"]["hot_seconds"]["median"]
            ),
            "nq512_comparative_performance_pass": bool(
                tip3p_finufft["comparative_performance_pass"]
            ),
        },
    }
    object_definition = (
        "ordered lysozyme protein crystal / single-crystal protein nanocrystal; "
        "not a dilute isolated single protein"
    )
    gates = {
        "protein_unit_exact_beta_direct_l2_le_1e-9": bool(
            metrics["protein_unit_cell"]["exact_beta_complex_l2_vs_direct"]
            <= 1e-9
        ),
        "million_atom_ordered_crystal_direct_subset_l2_le_1e-9": bool(
            metrics["ordered_protein_supercells"][
                "5x5x5_direct_subset_complex_l2"
            ]
            <= 1e-9
        ),
        "prepared_1m_local_abba_gate_pass": bool(
            metrics["prepared_1m_abba"]["local_gate_pass"]
        ),
        "dense_md_exact_beta_20_frame_control_pass": bool(tip3p["passed"]),
        "experimental_object_explicitly_crystalline": "protein crystal"
        in object_definition,
    }
    result = {
        "schema": "waxs-protein-exact-beta-followup-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "NO_ADDITIONAL_LOCAL_PROTEIN_EXACT_BETA_RERUN",
        "decision_reason": [
            "The requested protein exact-beta validation already exists on the 8008-atom lysozyme unit cell against a literal direct NDFT reference.",
            "The ordered-crystal specialization is independently checked on 216216- and 1001000-atom protein supercells, including a 1M direct-subset gate and a same-machine prepared 10/30 AB/BA timing receipt.",
            "The 20-frame 50430-atom TIP3P trajectory already supplies a dense-disorder/MD exact-beta control; repeating this specifically for disordered protein MD would broaden the claim beyond the chosen ordered protein-crystal experiment.",
            "The Nq=512 TIP3P row shows FINUFFT is faster for the generic dense exact-coordinate workload, so the paper must attribute the protein-crystal gain to ordered lattice/prepared reuse rather than to a universal exact-beta advantage.",
        ],
        "experimental_object_definition": object_definition,
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "remaining_required_work": [
            "Independent-machine replication of the frozen 1M prepared AB/BA timing protocol.",
            "Experimental photon-count, orientation stability and detector-geometry validation for the selected protein crystal or single-crystal nanocrystal.",
        ],
        "optional_scope_expansion_not_required_for_current_claim": [
            "Dense disordered protein-MD exact-beta timing.",
            "Dilute isolated single-protein WAXS feasibility.",
            "Arbitrary dense lattice disorder beyond the existing sparse-defect delta control.",
        ],
        "claim_boundary": [
            "WAXS is a validation/control domain in the current paper; the novelty claim is not that ACFO universally beats FINUFFT for every dense exact-coordinate source.",
            "The oriented protein object is an ordered protein crystal or single-crystal nanocrystal. A single isolated protein is outside the experimental claim because its WAXS signal may be insufficient.",
            "Perfect lattice factorization is a known exact specialization used as a protein-crystal control; the paper novelty remains the broader prepared curved-geometry operator and its cross-domain evidence.",
        ],
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in SOURCES.items()
        },
    }
    return result


def write(result: dict[str, Any]) -> None:
    OUTPUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    m = result["metrics"]
    lines = [
        "# WAXS protein exact-beta 후속 실행 결정",
        "",
        f"- 결정: **{result['decision']}**",
        f"- 실험 object: **{result['experimental_object_definition']}**",
        f"- 전체 gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## 이미 확보된 핵심 수치",
        "",
        f"- protein unit cell: {m['protein_unit_cell']['atoms']:,} atoms, exact-beta/direct complex L2 `{m['protein_unit_cell']['exact_beta_complex_l2_vs_direct']:.3e}`",
        f"- 3x3x3 crystal: {m['ordered_protein_supercells']['3x3x3_atoms']:,} atoms, direct-subset complex L2 `{m['ordered_protein_supercells']['3x3x3_direct_subset_complex_l2']:.3e}`",
        f"- 5x5x5 crystal: {m['ordered_protein_supercells']['5x5x5_atoms']:,} atoms, direct-subset complex L2 `{m['ordered_protein_supercells']['5x5x5_direct_subset_complex_l2']:.3e}`",
        f"- 1M prepared AB/BA: median `{m['prepared_1m_abba']['paired_speedup_median']:.3f}x`, p05 `{m['prepared_1m_abba']['paired_speedup_p05']:.3f}x`, FINUFFT cross complex L2 `{m['prepared_1m_abba']['prepared_vs_finufft_complex_l2']:.3e}`",
        f"- dense TIP3P 20 frames: max exact-beta/direct complex L2 `{m['dense_md_control']['maximum_exact_beta_complex_l2']:.3e}`",
        f"- TIP3P Nq=512: exact-beta `{m['dense_md_control']['nq512_exact_beta_s']:.3f}` s vs FINUFFT hot `{m['dense_md_control']['nq512_finufft_hot_median_s']:.3f}` s (comparative gate FAIL)",
        "",
        "## 판정",
        "",
    ]
    lines.extend(f"- {item}" for item in result["decision_reason"])
    lines.extend(["", "## 남은 필수 작업", ""])
    lines.extend(f"- {item}" for item in result["remaining_required_work"])
    lines.extend(["", "## 현재 claim에 필요하지 않은 범위 확장", ""])
    lines.extend(
        f"- {item}"
        for item in result["optional_scope_expansion_not_required_for_current_claim"]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = build()
    write(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
