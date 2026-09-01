from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_memory_status_has_bounded_fraction() -> None:
    module = load_module(
        "q_sampling_case_for_test",
        "scripts/benchmark_protein_lattice_q_sampling_case.py",
    )
    status = module.physical_memory_status()
    assert set(status) == {"total_mib", "available_mib", "available_fraction"}
    if status["available_fraction"] is not None:
        assert 0.0 <= status["available_fraction"] <= 1.0
        assert status["total_mib"] > 0.0


def test_highq_linear_block_holdout_gate_and_prediction() -> None:
    module = load_module(
        "highq_threshold_for_test",
        "scripts/benchmark_protein_lattice_highq_threshold_strategy.py",
    )
    q_centers = np.linspace(6.7, 8.0, 20)
    block_rows = [
        {
            "q_min_inv_angstrom": float(q - 0.01),
            "q_max_inv_angstrom": float(q + 0.01),
            "block_wall_seconds": float(-6.5 + 2.1 * q),
            "target_count": 1728,
        }
        for q in q_centers
    ]
    case = {"timing_seconds": {"finufft": {"block_rows": block_rows}}}
    model = module.block_holdout_model([case], 0.20)
    assert model["passed"]
    assert model["holdout_relative_error"] < 1e-12

    predicted, block_count = module.predict_resolution_finufft_seconds(
        nq=32,
        q_min=6.7,
        q_max=8.0,
        q_block=2,
        model=model,
    )
    assert block_count == 16
    assert predicted > 0.0


def test_highq_constant_model_is_not_silently_emitted() -> None:
    module = load_module(
        "highq_threshold_gate_for_test",
        "scripts/benchmark_protein_lattice_highq_threshold_strategy.py",
    )
    case = {
        "timing_seconds": {
            "finufft": {
                "block_rows": [
                    {
                        "q_min_inv_angstrom": 6.7,
                        "q_max_inv_angstrom": 6.8,
                        "block_wall_seconds": 8.0,
                        "target_count": 1728,
                    }
                    for _ in range(4)
                ]
            }
        }
    }
    model = module.block_holdout_model([case], 0.20)
    assert not model["available"]
    assert not model["passed"]
