from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_waxs_protein_exact_beta_followup_decision import build  # noqa: E402


def test_followup_decision_uses_existing_protein_crystal_evidence() -> None:
    result = build()
    assert result["passed"] is True
    assert result["decision"] == "NO_ADDITIONAL_LOCAL_PROTEIN_EXACT_BETA_RERUN"
    assert "protein crystal" in result["experimental_object_definition"]
    assert result["metrics"]["protein_unit_cell"]["atoms"] == 8008
    assert (
        result["metrics"]["ordered_protein_supercells"]["5x5x5_atoms"]
        == 1_001_000
    )
    assert (
        result["metrics"]["dense_md_control"][
            "nq512_comparative_performance_pass"
        ]
        is False
    )
