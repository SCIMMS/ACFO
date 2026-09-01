from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        path = SCRIPTS / "benchmark_odt_banded_cartesian_final_packed.py"
        spec = importlib.util.spec_from_file_location("odt_banded_final_packed", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_final_packed_integration_parser_freezes_candidate_and_reference() -> None:
    module = load_module()
    args = module.parser().parse_args([])

    assert args.variant == "banded_inner96_outer64"
    assert args.h_cutoff == 28
    assert args.reference_h_cutoff == 36
    assert args.prune_axis_l0 is True
    assert args.axial_lowrank_rank == 16
    assert args.ring_adaptive_l_packed_threshold == 1e-6
    assert args.forward_mode == "auto"
    assert args.adjoint_mode == "auto"
    assert args.selected_slices == "1,8"

    reference = module.reference_args(args)
    assert reference.h_cutoff == 36
    assert reference.prune_axis_l0 is False
    assert reference.axial_lowrank_rank == 0
    assert reference.ring_adaptive_l_packed_threshold == 0.0


def test_final_packed_candidate_contract_is_explicit() -> None:
    module = load_module()
    args = module.parser().parse_args([])
    contract = module.candidate_contract(args)

    assert contract == {
        "h_cutoff": 28,
        "prune_axis_l0": True,
        "axial_lowrank_rank": 16,
        "ring_adaptive_l_packed_threshold": 1e-6,
        "radial_block_size": 32,
        "illumination_block_size": 4,
        "forward_mode": "auto",
        "adjoint_mode": "auto",
    }
