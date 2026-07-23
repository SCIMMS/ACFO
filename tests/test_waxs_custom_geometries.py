from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_waxs_custom_geometries",
    ROOT / "scripts" / "validate_waxs_custom_geometries.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(name: str, qmax: float, distance: float) -> dict[str, object]:
    return {
        "name": name,
        "all_finite": True,
        "complex_l2": 1e-7,
        "intensity_l2": 2e-7,
        "ring_l2": 3e-7,
        "active_fraction": 0.5,
        "outer_ring_active_fraction": 0.25,
        "qmax_inv_angstrom": qmax,
        "distance_mm": distance,
        "finufft_plan_mode": "memory_safe_q_blocked_setup_per_evaluation",
    }


def test_geometry_gates_require_accuracy_and_distinct_envelopes() -> None:
    gates = MODULE.build_gates(
        [_row("wide", 8.0, 80.0), _row("narrow", 1.2, 250.0)]
    )
    assert all(gates.values())


def test_receipt_contract_freezes_numerical_controls() -> None:
    case = MODULE.CASES[0]
    receipt = {
        "schema": "protein-nanocrystal-finufft-fair-v2",
        "source_mode": "same_binned",
        "qmin": case["qmin"],
        "qmax": case["qmax"],
        "nq": case["nq"],
        "wavelength_nm": case["wavelength_nm"],
        "detector_active_width_mm": case["active_width_mm"],
        "detector_active_height_mm": case["active_height_mm"],
        "detector_distance_mm": case["distance_mm"],
        "harmonic_margin": 32,
        "r_dependent_margin": 32,
        "finufft_q_block_size": 2,
    }
    assert MODULE.receipt_matches_case(receipt, case)
    receipt["r_dependent_margin"] = 16
    assert not MODULE.receipt_matches_case(receipt, case)
