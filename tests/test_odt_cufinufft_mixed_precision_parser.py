from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_cufinufft_gpu_baseline import parser  # noqa: E402


def test_cufinufft_dtype_defaults_to_legacy_same_precision() -> None:
    args = parser().parse_args([])
    assert args.dtype == "complex64"
    assert args.cufinufft_dtype == "same"


def test_cufinufft_dtype_can_be_promoted_independently() -> None:
    args = parser().parse_args(["--dtype", "complex64", "--cufinufft-dtype", "complex128"])
    assert args.dtype == "complex64"
    assert args.cufinufft_dtype == "complex128"
