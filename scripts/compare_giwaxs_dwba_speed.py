from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_giwaxs_dwba_multilayer import (  # noqa: E402
    binned_dwba_direct_amplitude,
    build_distorted_wave_stack,
    build_prepared_dwba_geometry,
    build_prepared_dwba_miller_geometry,
    direct_dwba_atom_amplitude,
    dwba_field_product_grid,
    execute_prepared_dwba_geometry,
    execute_prepared_dwba_miller_geometry,
    finufft_dwba_channel_amplitude,
    make_synthetic_multilayer,
    median_time,
)
from benchmark_giwaxs_prepared_operator import (  # noqa: E402
    binned_direct_amplitude as kinematic_binned_direct_amplitude,
    build_prepared_giwaxs_geometry,
    build_prepared_giwaxs_miller_geometry,
    execute_prepared_giwaxs_geometry,
    execute_prepared_giwaxs_miller_geometry,
    make_giwaxs_detector,
    summarize_detector,
)
from waxs_cake import make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_sci(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3e}"


def _fmt_ratio(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}x"


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _time_or_skip(label: str, func, repeats: int) -> tuple[Any | None, float | None, list[float], str | None]:
    try:
        value, seconds, times = median_time(func, repeats)
    except Exception as exc:  # pragma: no cover - optional compiled path
        return None, None, [], f"{label} skipped: {type(exc).__name__}: {exc}"
    return value, seconds, times, None


def run_case(args: argparse.Namespace, detector, n_atoms: int, case_index: int) -> dict[str, Any]:
    coords, layer_ids, z_edges = make_synthetic_multilayer(
        n_atoms=n_atoms,
        n_layers=args.n_layers,
        radius_nm=args.radius_nm,
        layer_thickness_nm=args.layer_thickness_nm,
        seed=args.seed + case_index,
    )
    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            element_indices=layer_ids,
            n_elements=args.n_layers,
            n_r=args.n_r,
            n_z=args.n_z,
            n_phi=args.n_phi,
            r_max=args.radius_nm,
            z_range=(float(z_edges[0]), float(z_edges[-1])),
            backend=args.hist_backend,
            hist_dtype=np.dtype(args.hist_dtype),
        ),
        args.repeats,
    )
    stack = build_distorted_wave_stack(
        detector,
        z_edges,
        critical_angle_start_deg=args.critical_angle_start_deg,
        critical_angle_step_deg=args.critical_angle_step_deg,
        absorption_imag=args.absorption_imag,
        reflectivity_scale=args.reflectivity_scale,
        beta_loss_per_layer=args.beta_loss_per_layer,
    )
    field_grid, field_s, field_times = median_time(
        lambda: dwba_field_product_grid(stack, binned.z_centers),
        args.repeats,
    )

    qx, qy, qz = detector.qx, detector.qy, detector.qz
    dwba_binned, dwba_binned_s, dwba_binned_times = median_time(
        lambda: binned_dwba_direct_amplitude(
            binned,
            field_grid,
            qx,
            qy,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )
    dwba_geometry, dwba_build_s, dwba_build_times = median_time(
        lambda: build_prepared_dwba_geometry(
            binned,
            qx,
            qy,
            max_mode=args.max_mode,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
        ),
        args.repeats,
    )
    dwba_prepared, dwba_hot_s, dwba_hot_times = median_time(
        lambda: execute_prepared_dwba_geometry(
            binned,
            dwba_geometry,
            field_grid,
            detector.shape,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )

    dwba_miller_geometry, dwba_miller_build_s, dwba_miller_build_times, dwba_miller_error = _time_or_skip(
        "dwba_miller_build",
        lambda: build_prepared_dwba_miller_geometry(
            binned,
            qx,
            qy,
            max_mode=args.max_mode,
            extra_order=args.miller_extra_order,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
            complex_dtype=args.miller_complex_dtype,
        ),
        args.repeats,
    )
    dwba_miller = None
    dwba_miller_hot_s = None
    dwba_miller_hot_times: list[float] = []
    if dwba_miller_geometry is not None:
        dwba_miller, dwba_miller_hot_s, dwba_miller_hot_times, run_error = _time_or_skip(
            "dwba_miller_hot",
            lambda: execute_prepared_dwba_miller_geometry(
                binned,
                dwba_miller_geometry,
                stack,
                detector.shape,
            ),
            args.repeats,
        )
        dwba_miller_error = dwba_miller_error or run_error

    dwba_atom = None
    dwba_atom_s = None
    dwba_atom_times: list[float] = []
    if n_atoms <= args.direct_atom_limit:
        dwba_atom, dwba_atom_s, dwba_atom_times = median_time(
            lambda: direct_dwba_atom_amplitude(
                coords,
                layer_ids,
                stack,
                qx,
                qy,
                target_chunk=args.target_chunk,
            ),
            max(1, min(args.repeats, 3)),
        )

    dwba_finufft = None
    dwba_finufft_s = None
    dwba_finufft_times: list[float] = []
    finufft_error = None
    if not args.skip_finufft:
        dwba_finufft, dwba_finufft_s, dwba_finufft_times, finufft_error = _time_or_skip(
            "dwba_finufft_channel",
            lambda: finufft_dwba_channel_amplitude(
                coords,
                layer_ids,
                stack,
                qx,
                qy,
                eps=args.finufft_eps,
                imag_tol=args.finufft_imag_tol,
            ),
            max(1, min(args.repeats, 3)),
        )

    kin_binned, kin_binned_s, kin_binned_times = median_time(
        lambda: kinematic_binned_direct_amplitude(
            binned,
            qx,
            qy,
            qz,
            target_chunk=args.target_chunk,
        ),
        args.repeats,
    )
    kin_dense_geometry, kin_dense_build_s, kin_dense_build_times = median_time(
        lambda: build_prepared_giwaxs_geometry(
            binned,
            qx,
            qy,
            qz,
            max_mode=args.max_mode,
        ),
        args.repeats,
    )
    kin_dense, kin_dense_hot_s, kin_dense_hot_times = median_time(
        lambda: execute_prepared_giwaxs_geometry(
            binned,
            kin_dense_geometry,
            detector.shape,
        ),
        args.repeats,
    )

    kin_miller_geometry, kin_miller_build_s, kin_miller_build_times, miller_error = _time_or_skip(
        "kinematic_miller_build",
        lambda: build_prepared_giwaxs_miller_geometry(
            binned,
            qx,
            qy,
            qz,
            max_mode=args.max_mode,
            extra_order=args.miller_extra_order,
            enable_mode_pruning=not args.disable_mode_pruning,
            mode_pruning_margin=args.mode_pruning_margin,
            mode_pruning_bin_size=args.mode_pruning_bin_size,
            enable_qz_reduction=not args.disable_qz_reduction,
            precompute_kernel=args.precompute_pruned_kernel,
            complex_dtype=args.miller_complex_dtype,
        ),
        args.repeats,
    )
    kin_miller = None
    kin_miller_hot_s = None
    kin_miller_hot_times: list[float] = []
    if kin_miller_geometry is not None:
        kin_miller, kin_miller_hot_s, kin_miller_hot_times, run_error = _time_or_skip(
            "kinematic_miller_hot",
            lambda: execute_prepared_giwaxs_miller_geometry(
                binned,
                kin_miller_geometry,
                detector.shape,
                source_backend=args.source_backend,
            ),
            args.repeats,
        )
        miller_error = miller_error or run_error

    dwba_binned_i = intensity(dwba_binned)
    dwba_prepared_i = intensity(dwba_prepared)
    dwba_miller_i = None if dwba_miller is None else intensity(dwba_miller)
    dwba_atom_i = None if dwba_atom is None else intensity(dwba_atom)
    dwba_finufft_i = None if dwba_finufft is None else intensity(dwba_finufft)
    kin_binned_i = intensity(kin_binned)
    kin_dense_i = intensity(kin_dense)
    kin_miller_i = None if kin_miller is None else intensity(kin_miller)
    hist = np.asarray(binned.hist)
    row = {
        "atoms": int(n_atoms),
        "layers": int(args.n_layers),
        "targets": int(qx.size),
        "grid": [int(args.n_r), int(args.n_z), int(args.n_phi)],
        "max_mode": None if args.max_mode is None else int(args.max_mode),
        "dwba_prepared_mode_pruning": bool(dwba_geometry.mode_pruning),
        "dwba_prepared_requested_max_mode": int(dwba_geometry.requested_max_mode),
        "dwba_prepared_active_max_mode": int(dwba_geometry.max_mode),
        "dwba_prepared_cutoff_min": int(dwba_geometry.cutoff_min),
        "dwba_prepared_cutoff_mean": float(dwba_geometry.cutoff_mean),
        "dwba_prepared_mode_work_fraction": float(dwba_geometry.mode_work_fraction),
        "active_bin_count": int(np.count_nonzero(hist)),
        "active_bin_fraction": float(np.count_nonzero(hist) / hist.size),
        "hist_s": hist_s,
        "field_recurrence_s": field_s,
        "dwba_atom_direct_s": dwba_atom_s,
        "dwba_binned_direct_s": dwba_binned_s,
        "dwba_prepared_build_s": dwba_build_s,
        "dwba_prepared_hot_s": dwba_hot_s,
        "dwba_miller_build_s": dwba_miller_build_s,
        "dwba_miller_hot_s": dwba_miller_hot_s,
        "dwba_miller_error": dwba_miller_error,
        "dwba_finufft_channel_s": dwba_finufft_s,
        "finufft_error": finufft_error,
        "kinematic_binned_direct_s": kin_binned_s,
        "kinematic_dense_build_s": kin_dense_build_s,
        "kinematic_dense_hot_s": kin_dense_hot_s,
        "kinematic_miller_build_s": kin_miller_build_s,
        "kinematic_miller_hot_s": kin_miller_hot_s,
        "miller_error": miller_error,
        "dwba_prepared_intensity_rel_l2_vs_binned_direct": relative_l2(
            dwba_prepared_i,
            dwba_binned_i,
        ),
        "dwba_miller_intensity_rel_l2_vs_binned_direct": None
        if dwba_miller_i is None
        else relative_l2(dwba_miller_i, dwba_binned_i),
        "dwba_miller_intensity_rel_l2_vs_dense_prepared": None
        if dwba_miller_i is None
        else relative_l2(dwba_miller_i, dwba_prepared_i),
        "dwba_binned_intensity_rel_l2_vs_atom_direct": None
        if dwba_atom_i is None
        else relative_l2(dwba_binned_i, dwba_atom_i),
        "dwba_finufft_intensity_rel_l2_vs_atom_direct": None
        if dwba_finufft_i is None or dwba_atom_i is None
        else relative_l2(dwba_finufft_i, dwba_atom_i),
        "dwba_prepared_intensity_rel_l2_vs_finufft": None
        if dwba_finufft_i is None
        else relative_l2(dwba_prepared_i, dwba_finufft_i),
        "dwba_miller_intensity_rel_l2_vs_finufft": None
        if dwba_finufft_i is None or dwba_miller_i is None
        else relative_l2(dwba_miller_i, dwba_finufft_i),
        "kinematic_dense_intensity_rel_l2_vs_binned_direct": relative_l2(
            kin_dense_i,
            kin_binned_i,
        ),
        "kinematic_miller_intensity_rel_l2_vs_binned_direct": None
        if kin_miller_i is None
        else relative_l2(kin_miller_i, kin_binned_i),
        "dwba_prepared_speedup_vs_atom_direct": _ratio(dwba_atom_s, dwba_hot_s),
        "dwba_prepared_speedup_vs_binned_direct": _ratio(dwba_binned_s, dwba_hot_s),
        "dwba_prepared_plus_field_speedup_vs_binned_direct": _ratio(
            dwba_binned_s,
            field_s + dwba_hot_s,
        ),
        "dwba_prepared_speedup_vs_finufft": _ratio(dwba_finufft_s, dwba_hot_s),
        "dwba_prepared_plus_field_speedup_vs_finufft": _ratio(
            dwba_finufft_s,
            field_s + dwba_hot_s,
        ),
        "dwba_miller_speedup_vs_atom_direct": _ratio(dwba_atom_s, dwba_miller_hot_s),
        "dwba_miller_speedup_vs_binned_direct": _ratio(dwba_binned_s, dwba_miller_hot_s),
        "dwba_miller_speedup_vs_dense_prepared": _ratio(dwba_hot_s, dwba_miller_hot_s),
        "dwba_miller_speedup_vs_finufft": _ratio(dwba_finufft_s, dwba_miller_hot_s),
        "dwba_overhead_vs_kinematic_dense_hot": _ratio(dwba_hot_s, kin_dense_hot_s),
        "dwba_miller_overhead_vs_kinematic_miller_hot": _ratio(
            dwba_miller_hot_s,
            kin_miller_hot_s,
        ),
        "dwba_overhead_vs_kinematic_miller_hot": _ratio(dwba_hot_s, kin_miller_hot_s),
        "kinematic_miller_speedup_vs_kinematic_binned_direct": _ratio(
            kin_binned_s,
            kin_miller_hot_s,
        ),
        "times": {
            "hist": hist_times,
            "field_recurrence": field_times,
            "dwba_atom_direct": dwba_atom_times,
            "dwba_binned_direct": dwba_binned_times,
            "dwba_prepared_build": dwba_build_times,
            "dwba_prepared_hot": dwba_hot_times,
            "dwba_miller_build": dwba_miller_build_times,
            "dwba_miller_hot": dwba_miller_hot_times,
            "dwba_finufft_channel": dwba_finufft_times,
            "kinematic_binned_direct": kin_binned_times,
            "kinematic_dense_build": kin_dense_build_times,
            "kinematic_dense_hot": kin_dense_hot_times,
            "kinematic_miller_build": kin_miller_build_times,
            "kinematic_miller_hot": kin_miller_hot_times,
        },
    }
    print(
        "{atoms}: DWBA atom={atom} binned={binned:.4f}s dense={prepared:.4f}s "
        "miller_wrapper={dwba_miller} finufft={finufft} dense/direct={speed} "
        "dense/finufft={fin_speed} dense_mode_work={mode_work:.1%} kinematic_miller={miller} "
        "DWBA_miller/kinematic={overhead}".format(
            atoms=n_atoms,
            atom="n/a" if dwba_atom_s is None else f"{dwba_atom_s:.4f}s",
            binned=dwba_binned_s,
            prepared=dwba_hot_s,
            dwba_miller="n/a" if dwba_miller_hot_s is None else f"{dwba_miller_hot_s:.4f}s",
            finufft="n/a" if dwba_finufft_s is None else f"{dwba_finufft_s:.4f}s",
            speed=_fmt_ratio(row["dwba_prepared_speedup_vs_binned_direct"]),
            fin_speed=_fmt_ratio(row["dwba_prepared_speedup_vs_finufft"]),
            mode_work=row["dwba_prepared_mode_work_fraction"],
            miller="n/a" if kin_miller_hot_s is None else f"{kin_miller_hot_s:.4f}s",
            overhead=_fmt_ratio(row["dwba_miller_overhead_vs_kinematic_miller_hot"]),
        )
    )
    return row


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# GIWAXS DWBA Speed Comparison",
        "",
        "Same synthetic multilayer source and detector are used for both comparisons.",
            "The DWBA rows compare equal-physics scattering paths in the propagating",
            "above-critical field-correction regime; the kinematic rows show how much",
            "overhead remains relative to the older non-DWBA prepared GIWAXS path.",
        "",
        "## Detector",
        "",
        "| field | value |",
        "|---|---:|",
    ]
    for key, value in summary["detector"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Equal-Physics DWBA Comparison",
            "",
            "| atoms | grid | targets | field recurrence s | atom direct s | binned direct s | dense cached DWBA s | dense mode work | C++ channel wrapper s | dense speedup vs atom | dense speedup vs binned | dense intensity L2 | C++ wrapper intensity L2 | binned-vs-atom intensity L2 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        grid = "x".join(str(item) for item in row["grid"])
        lines.append(
            "| {atoms} | `{grid}` | {targets} | `{field}` | `{atom}` | `{binned}` | "
            "`{prepared}` | {mode_work} | `{miller}` | {speed_atom} | {speed_binned} | "
            "{prep_l2} | {miller_l2} | {atom_l2} |".format(
                atoms=row["atoms"],
                grid=grid,
                targets=row["targets"],
                field=_fmt(row["field_recurrence_s"]),
                atom=_fmt(row["dwba_atom_direct_s"]),
                binned=_fmt(row["dwba_binned_direct_s"]),
                prepared=_fmt(row["dwba_prepared_hot_s"]),
                mode_work=_fmt_pct(row["dwba_prepared_mode_work_fraction"]),
                miller=_fmt(row["dwba_miller_hot_s"]),
                speed_atom=_fmt_ratio(row["dwba_prepared_speedup_vs_atom_direct"]),
                speed_binned=_fmt_ratio(row["dwba_prepared_speedup_vs_binned_direct"]),
                prep_l2=_fmt_sci(row["dwba_prepared_intensity_rel_l2_vs_binned_direct"]),
                miller_l2=_fmt_sci(row["dwba_miller_intensity_rel_l2_vs_binned_direct"]),
                atom_l2=_fmt_sci(row["dwba_binned_intensity_rel_l2_vs_atom_direct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Generic FINUFFT Channel Baseline",
            "",
            "The four DWBA channels per layer are evaluated as generic FINUFFT type-3",
            "nonuniform Fourier sums. This is an exact same-physics baseline only for",
            "the real-effective-qz run used here.",
            "",
            "| atoms | FINUFFT channel s | dense cached DWBA s | C++ channel wrapper s | dense speedup vs FINUFFT | wrapper speedup vs FINUFFT | FINUFFT intensity L2 vs atom | dense intensity L2 vs FINUFFT |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {atoms} | `{finufft}` | `{prepared}` | `{miller}` | {speed} | "
            "{wrapper_speed} | {fin_l2} | {prep_l2} |".format(
                atoms=row["atoms"],
                finufft=_fmt(row["dwba_finufft_channel_s"]),
                prepared=_fmt(row["dwba_prepared_hot_s"]),
                miller=_fmt(row["dwba_miller_hot_s"]),
                speed=_fmt_ratio(row["dwba_prepared_speedup_vs_finufft"]),
                wrapper_speed=_fmt_ratio(row["dwba_miller_speedup_vs_finufft"]),
                fin_l2=_fmt_sci(row["dwba_finufft_intensity_rel_l2_vs_atom_direct"]),
                prep_l2=_fmt_sci(row["dwba_prepared_intensity_rel_l2_vs_finufft"]),
            )
        )
    lines.extend(
        [
            "",
            "## Existing Kinematic GIWAXS Comparison",
            "",
            "| atoms | kinematic binned direct s | kinematic dense hot s | kinematic Miller hot s | Miller speedup vs kinematic binned | dense DWBA / kinematic Miller | Miller DWBA / kinematic Miller | kinematic Miller intensity L2 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {atoms} | `{kin_binned}` | `{kin_dense}` | `{kin_miller}` | {kin_speed} | "
            "{dense_overhead} | {miller_overhead} | {miller_l2} |".format(
                atoms=row["atoms"],
                kin_binned=_fmt(row["kinematic_binned_direct_s"]),
                kin_dense=_fmt(row["kinematic_dense_hot_s"]),
                kin_miller=_fmt(row["kinematic_miller_hot_s"]),
                kin_speed=_fmt_ratio(row["kinematic_miller_speedup_vs_kinematic_binned_direct"]),
                dense_overhead=_fmt_ratio(row["dwba_overhead_vs_kinematic_miller_hot"]),
                miller_overhead=_fmt_ratio(row["dwba_miller_overhead_vs_kinematic_miller_hot"]),
                miller_l2=_fmt_sci(row["kinematic_miller_intensity_rel_l2_vs_binned_direct"]),
            )
        )
    skipped = [
        error
        for row in rows
        for error in (
            row.get("miller_error"),
            row.get("dwba_miller_error"),
            row.get("finufft_error"),
        )
        if error
    ]
    if skipped:
        lines.extend(["", "## Skipped Optional Paths", ""])
        lines.extend(f"- {item}" for item in skipped)
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Equal-physics speedup should be read from the DWBA table: atom direct and binned direct include the same propagating incident/exit field correction.",
            "- The best current path is the dense cached contraction with qR-based mode pruning: source harmonic coefficients are built once, then the hot loop is dominated by the remaining fixed detector/grid contraction.",
            "- The FINUFFT table is the closest generic Fourier baseline for real effective channel qz. Evanescent fields are intentionally outside this comparison and need a separate stabilized field provider.",
            "- The kinematic Miller row is not a DWBA simulator; it is the older optimized GIWAXS prepared operator without distorted fields.",
            "- The C++ channel wrapper is retained as an experiment, but it is slower here because it expands the four DWBA channels per layer through repeated qz-reduced Miller work.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{summary['config']['out']}`",
            f"- Markdown: `{summary['config']['summary_md']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DWBA multilayer scattering speed against direct and existing GIWAXS prepared baselines."
    )
    parser.add_argument("--atoms", type=int, nargs="+", default=[8000, 50000])
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--radius-nm", type=float, default=4.0)
    parser.add_argument("--layer-thickness-nm", type=float, default=2.0)
    parser.add_argument("--n-r", type=int, default=32)
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-phi", type=int, default=192)
    parser.add_argument("--max-mode", type=int, default=95)
    parser.add_argument("--wavelength-nm", type=float, default=0.15406)
    parser.add_argument("--alpha-i-deg", type=float, default=0.2)
    parser.add_argument("--alpha-f-min-deg", type=float, default=0.3)
    parser.add_argument("--alpha-f-max-deg", type=float, default=6.0)
    parser.add_argument("--n-alpha-f", type=int, default=8)
    parser.add_argument("--two-theta-min-deg", type=float, default=-6.0)
    parser.add_argument("--two-theta-max-deg", type=float, default=6.0)
    parser.add_argument("--n-two-theta", type=int, default=12)
    parser.add_argument("--critical-angle-start-deg", type=float, default=0.13)
    parser.add_argument("--critical-angle-step-deg", type=float, default=0.018)
    parser.add_argument("--absorption-imag", type=float, default=0.0)
    parser.add_argument("--reflectivity-scale", type=float, default=0.28)
    parser.add_argument("--beta-loss-per-layer", type=float, default=0.018)
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="numpy")
    parser.add_argument("--hist-dtype", default="float32")
    parser.add_argument("--target-chunk", type=int, default=96)
    parser.add_argument("--direct-atom-limit", type=int, default=10000)
    parser.add_argument("--miller-extra-order", type=int, default=64)
    parser.add_argument("--skip-finufft", action="store_true")
    parser.add_argument("--finufft-eps", type=float, default=1e-9)
    parser.add_argument("--finufft-imag-tol", type=float, default=1e-12)
    parser.add_argument("--source-backend", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--mode-pruning-margin", type=int, default=32)
    parser.add_argument("--mode-pruning-bin-size", type=int, default=1)
    parser.add_argument("--disable-mode-pruning", action="store_true")
    parser.add_argument("--disable-qz-reduction", action="store_true")
    parser.add_argument("--precompute-pruned-kernel", action="store_true")
    parser.add_argument(
        "--miller-complex-dtype",
        choices=["complex64", "complex128"],
        default="complex128",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_dwba_speed_comparison.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "giwaxs_dwba_speed_comparison.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = make_giwaxs_detector(
        wavelength_nm=args.wavelength_nm,
        alpha_i_deg=args.alpha_i_deg,
        alpha_f_min_deg=args.alpha_f_min_deg,
        alpha_f_max_deg=args.alpha_f_max_deg,
        n_alpha_f=args.n_alpha_f,
        two_theta_min_deg=args.two_theta_min_deg,
        two_theta_max_deg=args.two_theta_max_deg,
        n_two_theta=args.n_two_theta,
    )
    rows = [run_case(args, detector, atoms, index) for index, atoms in enumerate(args.atoms)]
    summary = {
        "config": _as_jsonable(vars(args) | {"out": args.out, "summary_md": args.summary_md}),
        "detector": summarize_detector(detector),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_as_jsonable(summary), indent=2), encoding="utf-8")
    write_summary(args.summary_md, summary)
    print(args.out)
    print(args.summary_md)


if __name__ == "__main__":
    main()
