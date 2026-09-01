from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_banded_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        path = SCRIPTS / "benchmark_odt_banded_detector.py"
        spec = importlib.util.spec_from_file_location("odt_banded_detector_options", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_banded_plan_options_preserve_legacy_defaults() -> None:
    module = load_banded_module()
    options = module.torch_plan_options(argparse.Namespace())

    assert options == {
        "low_memory_adjoint": True,
        "radial_block_size": 0,
        "illumination_block_size": 0,
        "forward_mode": "illumination-reduced",
        "adjoint_mode": "illumination-reduced",
        "prune_axis_l0": False,
        "axial_lowrank_rank": 0,
        "ring_adaptive_l_packed_threshold": 0.0,
    }


def test_banded_plan_options_forward_final_packed_controls() -> None:
    module = load_banded_module()
    options = module.torch_plan_options(
        argparse.Namespace(
            low_memory_adjoint=True,
            radial_block_size=32,
            illumination_block_size=4,
            forward_mode="auto",
            adjoint_mode="auto",
            prune_axis_l0=True,
            axial_lowrank_rank=16,
            ring_adaptive_l_packed_threshold=1e-6,
        )
    )

    assert options["prune_axis_l0"] is True
    assert options["axial_lowrank_rank"] == 16
    assert options["ring_adaptive_l_packed_threshold"] == 1e-6
    assert options["radial_block_size"] == 32
    assert options["illumination_block_size"] == 4
    assert options["forward_mode"] == "auto"
    assert options["adjoint_mode"] == "auto"
