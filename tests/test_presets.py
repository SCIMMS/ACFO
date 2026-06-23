from __future__ import annotations

import argparse

from waxs_cake import FAST_PRESET_NAMES, fast_preset_options
from waxs_cake.presets import apply_fast_preset


def test_production_fast_preset_options() -> None:
    options = fast_preset_options("production")

    assert "production" in FAST_PRESET_NAMES
    assert options["hist_backend"] == "cpp"
    assert options["hist_dtype"] == "float32"
    assert options["angle_lut_mode"] == "cubic"
    assert options["angle_lut_size"] == 32
    assert options["circular_backend"] == "cpp"
    assert options["complex_dtype"] == "auto"
    assert options["harmonic_bandlimit_margin"] is None


def test_apply_production_bandlimited_fast_preset() -> None:
    args = argparse.Namespace(
        fast_preset="production-bandlimited",
        hist_backend="numpy",
        hist_dtype="default",
        angle_lut_size=0,
        angle_lut_mode="nearest",
        circular_backend="auto",
        complex_dtype="complex128",
        q_block_size=8,
        harmonic_bandlimit_margin=None,
    )

    apply_fast_preset(args)

    assert args.hist_backend == "cpp"
    assert args.hist_dtype == "float32"
    assert args.angle_lut_size == 32
    assert args.angle_lut_mode == "cubic"
    assert args.circular_backend == "cpp"
    assert args.complex_dtype == "auto"
    assert args.q_block_size == 128
    assert args.harmonic_bandlimit_margin == 16
