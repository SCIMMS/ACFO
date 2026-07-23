from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_machine_environment_receipt_is_stable_and_source_bound() -> None:
    module = load_module(
        "prepared_waxs_environment_for_test",
        "scripts/collect_prepared_waxs_machine_environment.py",
    )
    first = module.build_receipt()
    second = module.build_receipt()
    assert first["schema"] == "prepared-waxs-machine-environment-v1"
    assert first["passed"] is True
    assert first["machine_fingerprint_sha256"] == second["machine_fingerprint_sha256"]
    assert len(first["machine_fingerprint_sha256"]) == 64
    assert first["source_sha256"]["benchmark_driver"] is not None
    assert first["source_sha256"]["cpp_solvers"] is not None
    assert first["source_sha256"]["exact_harmonic"] is not None


def test_same_machine_external_submission_is_rejected(tmp_path: Path) -> None:
    result = ROOT / "benchmark_results/protein_lattice_prepared_finufft_512_abba.json"
    environment = ROOT / "benchmark_results/local_prepared_waxs_machine_environment.json"
    output = tmp_path / "validation.json"
    summary = tmp_path / "validation.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_external_prepared_waxs_abba.py"),
            str(result),
            str(environment),
            "--local-result",
            str(result),
            "--local-environment",
            str(environment),
            "--output",
            str(output),
            "--summary-md",
            str(summary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["passed"] is False
    assert receipt["gates"]["machine_fingerprint_differs"] is False
    assert all(
        passed
        for gate, passed in receipt["gates"].items()
        if gate != "machine_fingerprint_differs"
    )


def test_reduced_suite_selects_available_torch_device() -> None:
    module = load_module(
        "reduced_release_suite_for_device_test",
        "scripts/run_acfo_ncs_reduced_release_suite.py",
    )
    cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)
    )
    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    assert module.choose_smoke_device(cuda_torch) == "cuda"
    assert module.choose_smoke_device(cpu_torch) == "cpu"
