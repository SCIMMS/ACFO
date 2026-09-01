from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_cufinufft_only import parser  # noqa: E402


def test_cufinufft_only_defaults_freeze_matched_full_protocol() -> None:
    args = parser().parse_args([])
    assert (args.n_r, args.n_z, args.n_beta) == (256, 256, 256)
    assert (args.cap_radial, args.cap_phi, args.ring_illum) == (256, 256, 120)
    assert args.cufinufft_dtype == "complex128"
    assert args.cufinufft_eps == 1e-7
    assert args.pair_warmups == 2
    assert args.pair_repeats == 5
    assert args.skip_native_prepared_adjoint is True
