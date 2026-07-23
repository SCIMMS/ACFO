from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_banded_cartesian_final_packed_full_timing import (  # noqa: E402
    parse_selected_counts,
    parser,
)


def test_full_timing_defaults_freeze_publication_scale() -> None:
    args = parser().parse_args([])
    assert args.selected_slices == "64,128,256"
    assert args.h_cutoff == 28
    assert args.prune_axis_l0 is True
    assert args.axial_lowrank_rank == 16
    assert args.ring_adaptive_l_packed_threshold == pytest.approx(1e-6)
    assert args.timing_warmups == 5
    assert args.timing_repeats == 30


def test_parse_selected_counts_rejects_invalid_or_duplicate_rows() -> None:
    assert parse_selected_counts("64,128,256", 256) == [64, 128, 256]
    with pytest.raises(ValueError):
        parse_selected_counts("0,8", 256)
    with pytest.raises(ValueError):
        parse_selected_counts("64,64", 256)
    with pytest.raises(ValueError):
        parse_selected_counts("257", 256)
