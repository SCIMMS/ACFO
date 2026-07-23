from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_acfo_claim_artifact_manifest.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("acfo_claim_manifest_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_manifest_maps_current_headline_evidence() -> None:
    builder = load_builder()
    manifest = builder.build_manifest()

    assert manifest["schema"] == "acfo-claim-artifact-manifest-v1"
    claims = {claim["claim_id"]: claim for claim in manifest["claims"]}
    assert set(claims) == {
        "waxs_detector_aware_local",
        "waxs_dense_exact_beta_highq",
        "aidt_gpu_resident_core",
        "odt_final_packed_operator",
        "odt_full_slab_update_throughput",
        "odt_cartesian_detector_remap_separate_branch",
        "general_curvature_frozen_holdout",
    }
    assert claims["waxs_detector_aware_local"]["metrics"]["ratio_of_medians_speedup"] > 1.9
    assert not claims["waxs_dense_exact_beta_highq"]["metrics"][
        "nq512_comparative_performance_pass"
    ]
    assert claims["aidt_gpu_resident_core"]["metrics"]["gpu_run_hz"] > 10.0
    assert claims["odt_final_packed_operator"]["metrics"]["worst_rel_l2_vs_h36"] <= 2e-6
    assert claims["odt_full_slab_update_throughput"]["metrics"]["full_update_hz"] > 8.0
    assert claims["general_curvature_frozen_holdout"]["metrics"]["passed_gate_count"] == 10
    gates = {gate["gate_id"]: gate for gate in manifest["open_gates"]}
    assert gates["odt_remap_final_packed_integration"]["status"] == "closed"
    assert gates["odt_cufinufft_matched_effective_error"]["result"][
        "acfo_speedup_vs_matched_cufinufft"
    ] > 300.0
    assert gates["odt_temporal_final_operator"]["result"][
        "warm_1_update_median_hot_hz"
    ] > 10.0
    assert gates["waxs_protein_exact_beta_followup"]["status"] == "closed_no_additional_rerun"


def test_claim_manifest_artifacts_are_hashed_and_open_gate_order_is_frozen() -> None:
    builder = load_builder()
    manifest = builder.build_manifest()

    for claim in manifest["claims"]:
        assert claim["artifacts"]
        for artifact in claim["artifacts"]:
            assert (ROOT / artifact["path"]).is_file()
            assert len(artifact["sha256"]) == 64
            assert artifact["bytes"] > 0

    gates = sorted(manifest["open_gates"], key=lambda item: item["priority"])
    assert gates[0]["gate_id"] == "odt_remap_final_packed_integration"
    assert gates[1]["gate_id"] == "odt_cufinufft_matched_effective_error"
    assert gates[2]["gate_id"] == "odt_temporal_final_operator"
    for artifact in manifest["gate_closure_artifacts"]:
        assert (ROOT / artifact["path"]).is_file()
        assert len(artifact["sha256"]) == 64
