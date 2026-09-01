from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_odt_cufinufft_matched_error import (  # noqa: E402
    parse_eps_values,
    parser,
    select_matched,
)


def test_parser_freezes_small_direct_production_contract() -> None:
    args = parser().parse_args([])
    assert (args.n_r, args.n_z, args.n_beta) == (24, 24, 64)
    assert (args.cap_radial, args.cap_phi, args.ring_illum) == (24, 64, 8)
    assert args.h_cutoff == 28
    assert args.axial_lowrank_rank == 16
    assert args.ring_adaptive_l_packed_threshold == pytest.approx(1e-6)
    assert args.q_subset_count == 4096
    assert args.dtype == "complex64"
    assert args.cufinufft_dtype == "complex64"


def test_eps_parser_orders_loose_to_strict() -> None:
    assert parse_eps_values("1e-6,1e-3,1e-4") == [1e-3, 1e-4, 1e-6]
    with pytest.raises(ValueError):
        parse_eps_values("0,1e-4")
    with pytest.raises(ValueError):
        parse_eps_values("1e-4,1e-4")


def test_matched_selection_prefers_loose_strict_directional_match() -> None:
    ours = {
        "forward_rel_l2_vs_direct": 2e-4,
        "adjoint_rel_l2_vs_direct": 3e-4,
        "worst_rel_l2_vs_direct": 3e-4,
    }
    rows = [
        {
            "eps": 1e-3,
            "forward_rel_l2_vs_direct": 1e-4,
            "adjoint_rel_l2_vs_direct": 4e-4,
            "worst_rel_l2_vs_direct": 4e-4,
        },
        {
            "eps": 3e-4,
            "forward_rel_l2_vs_direct": 1.5e-4,
            "adjoint_rel_l2_vs_direct": 2.5e-4,
            "worst_rel_l2_vs_direct": 2.5e-4,
        },
        {
            "eps": 1e-4,
            "forward_rel_l2_vs_direct": 1e-4,
            "adjoint_rel_l2_vs_direct": 1e-4,
            "worst_rel_l2_vs_direct": 1e-4,
        },
    ]
    matched = select_matched(rows, ours)
    assert matched["eps"] == pytest.approx(3e-4)
    assert matched["strict_directional_match_exists"] is True
