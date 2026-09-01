from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_banded_cartesian_temporal_warm_start import (  # noqa: E402
    aggregate,
    parser,
)


def test_temporal_parser_freezes_small_motion_final_operator_sequence() -> None:
    args = parser().parse_args([])
    assert args.frames == 8
    assert args.selected_n_z == 8
    assert args.updates_per_frame == "1,2,3,5"
    assert args.motion_fraction == 0.01
    assert args.phase_drift_rad == 0.02
    assert args.h_cutoff == 28
    assert args.axial_lowrank_rank == 16
    assert args.ring_adaptive_l_packed_threshold == 1e-6


def test_aggregate_adds_warm_cold_and_reference_ratios() -> None:
    rows = []
    for mode, updates, error in (
        ("warm_start", 1, 0.1),
        ("cold_start", 1, 0.2),
        ("reference", 20, 0.08),
    ):
        rows.append(
            {
                "mode": mode,
                "updates": updates,
                "frame": 1,
                "total_hot_s": 0.05,
                "object_rel_l2": error,
                "data_residual_rel_l2": error,
            }
        )
    summary = aggregate(rows)
    warm = next(row for row in summary if row["mode"] == "warm_start")
    assert warm["mean_object_error_vs_cold_ratio"] == 0.5
    assert warm["mean_object_error_vs_reference_ratio"] == 1.25
