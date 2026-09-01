from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_acfo_ncs_validation_closure_report import collect  # noqa: E402


def test_report_metrics_are_recomputed_from_frozen_evidence() -> None:
    metrics = collect()["metrics"]
    assert metrics["odt"]["probe"]["worst_operator_l2"] <= 2e-6
    assert metrics["odt"]["scale"]["64"]["steady_hz"] > 10.0
    assert metrics["odt"]["matched"]["speedup"] > 300.0
    assert metrics["odt"]["temporal"]["rows"][("warm_start", 1)][
        "median_hot_hz"
    ] > 10.0
    assert metrics["aidt"]["hz"] > 10.0
    assert metrics["waxs"]["protein_unit_cell"][
        "exact_beta_complex_l2_vs_direct"
    ] < 1e-9
    assert metrics["general_curvature"]["gates"] == 10
