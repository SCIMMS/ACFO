"""Reusable runtime presets for command-line benchmark scripts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FAST_PRESET_NAMES = ("none", "production", "production-bandlimited")

_FAST_PRESETS: Mapping[str, Mapping[str, Any]] = {
    "none": {},
    "production": {
        "hist_backend": "cpp",
        "hist_dtype": "float32",
        "angle_lut_size": 32,
        "angle_lut_mode": "cubic",
        "circular_backend": "cpp",
        "complex_dtype": "auto",
        "q_block_size": 128,
        "harmonic_bandlimit_margin": None,
    },
    "production-bandlimited": {
        "hist_backend": "cpp",
        "hist_dtype": "float32",
        "angle_lut_size": 32,
        "angle_lut_mode": "cubic",
        "circular_backend": "cpp",
        "complex_dtype": "auto",
        "q_block_size": 128,
        "harmonic_bandlimit_margin": 16,
    },
}


def fast_preset_options(name: str) -> dict[str, Any]:
    """Return a copy of the option macro for a named fast preset."""

    if name not in _FAST_PRESETS:
        raise ValueError(f"unknown fast preset {name!r}")
    return dict(_FAST_PRESETS[name])


def apply_fast_preset(args: Any) -> None:
    """Apply ``args.fast_preset`` to an ``argparse.Namespace`` in place."""

    name = getattr(args, "fast_preset", "none")
    for key, value in fast_preset_options(name).items():
        if hasattr(args, key):
            setattr(args, key, value)
