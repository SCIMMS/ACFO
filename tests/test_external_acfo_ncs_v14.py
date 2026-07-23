from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.run_external_acfo_ncs_validation_v14 import (
    build_step_plan,
    verify_manifest,
)
from scripts.validate_external_acfo_ncs_v14 import FILES, evaluate_run


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_step_plan_separates_quick_from_full(tmp_path: Path) -> None:
    quick = build_step_plan("python", tmp_path, mode="quick")
    full = build_step_plan("python", tmp_path, mode="full")
    quick_labels = [step.label for step in quick]
    full_labels = [step.label for step in full]
    assert quick_labels == [
        "environment",
        "pip_freeze",
        "build_cpp_extensions",
        "pytest",
        "odt_integrated_probe",
    ]
    assert "waxs_prepared_1m_abba" not in quick_labels
    assert "waxs_prepared_1m_abba" in full_labels
    assert "odt_matched_c128_full_pair5" in full_labels
    assert "odt_temporal_warm_start" in full_labels
    assert all(
        path.is_relative_to(tmp_path)
        for step in full
        for path in step.outputs
    )


def test_manifest_verification_detects_changes(tmp_path: Path) -> None:
    payload = b"frozen\n"
    (tmp_path / "a.txt").write_bytes(payload)
    write_json(
        tmp_path / "MANIFEST.json",
        {
            "schema": "acfo-ncs-validation-release-candidate-v2",
            "files": [
                {
                    "path": "a.txt",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        },
    )
    assert verify_manifest(tmp_path)["passed"]
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    receipt = verify_manifest(tmp_path)
    assert not receipt["passed"]
    assert receipt["mismatch_count"] == 1


def test_full_validator_accepts_frozen_local_artifacts_as_synthetic_external(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / FILES["manifest"], {"passed": True})
    write_json(
        tmp_path / FILES["environment"],
        {
            "schema": "acfo-ncs-v14-machine-environment-v1",
            "machine_fingerprint_sha256": "synthetic-external-machine",
            "runtime": {
                "packages": {"cupy-cuda12x": "14.1.1", "cufinufft": "2.5.1"}
            },
            "torch": {"cuda_available": True},
        },
    )
    write_json(tmp_path / FILES["build_ext"], {"passed": True})
    write_json(tmp_path / FILES["pytest"], {"passed": True})
    sources = {
        "odt_probe": "benchmark_results/odt_banded_cartesian_final_packed_probe.json",
        "waxs_prepared": "benchmark_results/protein_lattice_prepared_finufft_512_abba.json",
        "waxs_detector": "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq512_w10_r30_alternating.json",
        "odt_scale": "benchmark_results/odt_banded_cartesian_final_packed_full_timing.json",
        "odt_direct_c64": "benchmark_results/odt_cufinufft_matched_error_direct_subset.json",
        "odt_direct_c128": "benchmark_results/odt_cufinufft_matched_error_direct_subset_c128.json",
        "odt_same_dtype": "benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json",
        "odt_matched_full": "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json",
        "odt_temporal": "benchmark_results/odt_banded_cartesian_temporal_warm_start.json",
    }
    for name, source in sources.items():
        shutil.copy2(ROOT / source, tmp_path / FILES[name])
    result = evaluate_run(tmp_path, mode="full")
    assert result["package_smoke_pass"]
    assert result["functional_correctness_pass"]
    assert result["performance_replication_pass"]
    assert result["independent_machine_replication_pass"]
    assert result["publication_replication_pass"]
    assert result["execution_pass"]


def test_v14_builder_includes_one_command_and_latest_gate_files() -> None:
    from scripts import build_acfo_ncs_release_candidate_v14 as builder

    selected = {path.relative_to(ROOT).as_posix() for path in builder.selected_files()}
    assert "scripts/install_and_run_external_acfo_ncs_v14.ps1" in selected
    assert "scripts/run_external_acfo_ncs_validation_v14.py" in selected
    assert "scripts/validate_external_acfo_ncs_v14.py" in selected
    assert "benchmark_results/odt_banded_cartesian_temporal_warm_start.json" in selected
    assert "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json" in selected
    claim_manifest = json.loads(
        (ROOT / "benchmark_results/acfo_claim_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    claim_paths = {
        artifact["path"]
        for claim in claim_manifest["claims"]
        for artifact in claim["artifacts"]
    }
    claim_paths.update(
        artifact["path"] for artifact in claim_manifest["gate_closure_artifacts"]
    )
    assert claim_paths <= selected
    assert "-Mode full" in builder.RELEASE_README


def test_v14_installer_auto_selects_a_supported_python_runtime() -> None:
    installer = (
        ROOT / "scripts/install_and_run_external_acfo_ncs_v14.ps1"
    ).read_text(encoding="utf-8")
    assert '[string]$PythonTag = "auto"' in installer
    assert '@("3.12", "3.11", "3.13")' in installer
    assert 'No supported Python runtime found' in installer


def test_v14_installer_pins_and_verifies_cuda_torch() -> None:
    installer = (
        ROOT / "scripts/install_and_run_external_acfo_ncs_v14.ps1"
    ).read_text(encoding="utf-8")
    assert '[string]$TorchVersion = "2.12.1"' in installer
    assert 'https://download.pytorch.org/whl/cu126' in installer
    assert "torch.cuda.is_available()" in installer
    assert "CUDA-enabled PyTorch verification failed" in installer


def test_v14_installer_installs_cupy_toolkit_components_and_refreshes_receipts() -> None:
    installer = (
        ROOT / "scripts/install_and_run_external_acfo_ncs_v14.ps1"
    ).read_text(encoding="utf-8")
    assert '[string]$CuPyVersion = "14.1.1"' in installer
    assert 'cupy-cuda12x[ctk]' in installer
    assert "jit_probe=" in installer
    assert '$env:CUPY_CACHE_DIR = Join-Path $Root ".cupy_cache"' in installer
    assert '"--refresh-receipts"' in installer


def test_resume_can_refresh_environment_receipts_without_repeating_completed_work() -> None:
    runner = (
        ROOT / "scripts/run_external_acfo_ncs_validation_v14.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--refresh-receipts", action="store_true")' in runner
    assert 'step.label in {' in runner
    assert '"environment"' in runner
    assert '"pip_freeze"' in runner
