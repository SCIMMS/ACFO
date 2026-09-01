from __future__ import annotations

import os

# Fix native thread pools before importing NumPy or FINUFFT.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from typing import Callable

import finufft
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    BinnedStructure,
    PreparedAxisymmetricOperator,
    PreparedFinufftAxisymmetricReference,
    binned_structure_grid,
    binned_structure_sources,
    direct_axisymmetric_amplitude,
)
try:
    from scripts.validate_axisymmetric_harmonic_cutoff import (  # noqa: E402
        active_radius_max,
        matched_curvature_family,
    )
except ModuleNotFoundError:  # pragma: no cover
    from validate_axisymmetric_harmonic_cutoff import (  # type: ignore[no-redef]  # noqa: E402
        active_radius_max,
        matched_curvature_family,
    )
try:
    from scripts.validate_axisymmetric_manifold_discrete import make_validation_object  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from validate_axisymmetric_manifold_discrete import make_validation_object  # type: ignore[no-redef]  # noqa: E402


def make_dense_benchmark_object(
    n_phi: int,
    *,
    n_r: int = 6,
    n_z: int = 5,
) -> BinnedStructure:
    r_edges = np.linspace(0.0, 2.4, n_r + 1)
    z_edges = np.linspace(-1.5, 1.5, n_z + 1)
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    envelope = np.exp(-0.45 * r[:, None, None] ** 2 - 0.38 * z[None, :, None] ** 2)
    angular = (
        1.0
        + 0.28 * np.cos(3.0 * beta)[None, None, :]
        + 0.17j * np.sin(5.0 * beta)[None, None, :]
        + 0.11 * np.cos(9.0 * beta + 0.3)[None, None, :]
    )
    histogram = (envelope * angular * (2.0 * np.pi / n_phi))[None, ...].astype(
        np.complex128
    )
    return BinnedStructure(
        hist=histogram,
        r_centers=r,
        z_centers=z,
        beta_centers=beta,
        elements=("X",),
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )


def timing(
    function: Callable[[], object],
    *,
    repeats: int,
    warmup: int = 0,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = perf_counter()
            function()
            samples.append(perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "median_s": float(statistics.median(samples)),
        "min_s": float(min(samples)),
        "samples_s": [float(value) for value in samples],
    }


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((actual - reference).ravel()) / np.linalg.norm(reference.ravel()))


def benchmark_case(
    n_u: int,
    n_phi: int,
    *,
    x_product: float = 12.0,
    finufft_eps: float = 1e-9,
    direct_interaction_limit: int = 20_000_000,
) -> dict[str, object]:
    histogram = make_dense_benchmark_object(n_phi)
    r_active = active_radius_max(histogram)
    manifold = matched_curvature_family(x_product, r_active, n_u=n_u)["ellipsoid"]
    coords, elements = binned_structure_grid(histogram)
    if set(elements) != {"X"}:
        raise RuntimeError("benchmark expects one default element")
    source_weights = np.asarray(histogram.hist).ravel()
    target_count = n_u * n_phi
    interactions = int(coords.shape[0] * target_count)

    acfo_setup = timing(
        lambda: PreparedAxisymmetricOperator(histogram, manifold),
        repeats=3,
    )
    acfo = PreparedAxisymmetricOperator(histogram, manifold)
    object_prepare = timing(
        lambda: acfo.prepare_object(histogram.hist),
        repeats=7,
        warmup=1,
    )
    object_fourier = acfo.prepare_object(histogram.hist)
    acfo_apply = timing(
        lambda: acfo.apply_prepared_object(object_fourier),
        repeats=7,
        warmup=1,
    )
    acfo_forward = timing(
        lambda: acfo.forward(histogram.hist),
        repeats=5,
        warmup=1,
    )
    acfo_output = acfo.apply_prepared_object(object_fourier)

    finufft_setup = timing(
        lambda: PreparedFinufftAxisymmetricReference(
            coords,
            manifold,
            histogram.beta_centers,
            eps=finufft_eps,
            nthreads=1,
        ),
        repeats=2,
    )
    finufft_plan = PreparedFinufftAxisymmetricReference(
        coords,
        manifold,
        histogram.beta_centers,
        eps=finufft_eps,
        nthreads=1,
    )
    finufft_execute = timing(
        lambda: finufft_plan.execute(source_weights),
        repeats=5,
        warmup=1,
    )
    finufft_output = finufft_plan.execute(source_weights)

    direct = None
    direct_output = None
    if interactions <= direct_interaction_limit:
        direct = timing(
            lambda: direct_axisymmetric_amplitude(
                coords,
                manifold,
                histogram.beta_centers,
                source_weights=source_weights,
            ),
            repeats=2,
        )
        direct_output = direct_axisymmetric_amplitude(
            coords,
            manifold,
            histogram.beta_centers,
            source_weights=source_weights,
        )

    acfo_cold = (
        float(acfo_setup["median_s"])
        + float(object_prepare["median_s"])
        + float(acfo_apply["median_s"])
    )
    finufft_cold = float(finufft_setup["median_s"]) + float(finufft_execute["median_s"])
    row: dict[str, object] = {
        "n_u": n_u,
        "n_phi": n_phi,
        "sources": int(coords.shape[0]),
        "targets": target_count,
        "interactions": interactions,
        "acfo": {
            "geometry_setup": acfo_setup,
            "object_fft": object_prepare,
            "prepared_apply": acfo_apply,
            "forward_with_object_fft": acfo_forward,
            "cold_total_s": acfo_cold,
        },
        "finufft": {
            "plan_setup": finufft_setup,
            "execute": finufft_execute,
            "cold_total_s": finufft_cold,
            "relative_l2_vs_acfo": relative_l2(finufft_output, acfo_output),
        },
        "direct": None,
        "speedups": {
            "finufft_execute_over_acfo_apply": float(finufft_execute["median_s"])
            / float(acfo_apply["median_s"]),
            "finufft_cold_over_acfo_cold": finufft_cold / acfo_cold,
        },
    }
    if direct is not None and direct_output is not None:
        row["direct"] = {
            "time": direct,
            "relative_l2_vs_acfo": relative_l2(direct_output, acfo_output),
        }
        row["speedups"]["direct_over_acfo_apply"] = float(direct["median_s"]) / float(
            acfo_apply["median_s"]
        )
    else:
        row["direct_skip_reason"] = (
            f"{interactions} source-target interactions exceed limit {direct_interaction_limit}"
        )
    return row


def curvature_batch(
    count: int,
    r_active: float,
    *,
    n_u: int,
) -> list[AxisymmetricManifold]:
    names = ("sphere", "ellipsoid", "paraboloid", "spline")
    batch = []
    for index in range(count):
        family = matched_curvature_family(
            11.5 + 0.15 * index,
            r_active,
            n_u=n_u,
        )
        batch.append(family[names[index % len(names)]])
    return batch


def benchmark_curvature_count(
    count: int,
    *,
    n_u: int = 64,
    n_phi: int = 128,
    finufft_eps: float = 1e-9,
) -> dict[str, object]:
    histogram = make_dense_benchmark_object(n_phi)
    coords, _ = binned_structure_grid(histogram)
    source_weights = np.asarray(histogram.hist).ravel()
    manifolds = curvature_batch(count, active_radius_max(histogram), n_u=n_u)

    def one_run() -> dict[str, float]:
        object_start = perf_counter()
        object_fourier = np.fft.fft(histogram.hist, axis=-1)
        object_prepare_s = perf_counter() - object_start

        setup_start = perf_counter()
        operators = [
            PreparedAxisymmetricOperator(histogram, manifold)
            for manifold in manifolds
        ]
        acfo_setup_s = perf_counter() - setup_start
        apply_start = perf_counter()
        for operator in operators:
            operator.apply_prepared_object(object_fourier)
        acfo_apply_s = perf_counter() - apply_start

        nufft_setup_start = perf_counter()
        plans = [
            PreparedFinufftAxisymmetricReference(
                coords,
                manifold,
                histogram.beta_centers,
                eps=finufft_eps,
                nthreads=1,
            )
            for manifold in manifolds
        ]
        nufft_setup_s = perf_counter() - nufft_setup_start
        nufft_execute_start = perf_counter()
        for plan in plans:
            plan.execute(source_weights)
        nufft_execute_s = perf_counter() - nufft_execute_start
        return {
            "object_prepare_s": object_prepare_s,
            "acfo_geometry_setup_s": acfo_setup_s,
            "acfo_apply_s": acfo_apply_s,
            "finufft_plan_setup_s": nufft_setup_s,
            "finufft_execute_s": nufft_execute_s,
        }

    samples = [one_run() for _ in range(3)]
    medians = {
        key: float(statistics.median(sample[key] for sample in samples))
        for key in samples[0]
    }
    acfo_total = (
        medians["object_prepare_s"]
        + medians["acfo_geometry_setup_s"]
        + medians["acfo_apply_s"]
    )
    finufft_total = medians["finufft_plan_setup_s"] + medians["finufft_execute_s"]
    return {
        "curvature_count": count,
        "n_u": n_u,
        "n_phi": n_phi,
        "sources": int(coords.shape[0]),
        "targets_per_curvature": n_u * n_phi,
        "medians": medians,
        "acfo_total_s": acfo_total,
        "finufft_total_s": finufft_total,
        "finufft_over_acfo_total": finufft_total / acfo_total,
        "acfo_time_per_curvature_s": acfo_total / count,
        "finufft_time_per_curvature_s": finufft_total / count,
    }


def repeat_amortization(case: dict[str, object]) -> list[dict[str, object]]:
    acfo = case["acfo"]
    nufft = case["finufft"]
    acfo_setup = float(acfo["geometry_setup"]["median_s"])
    object_fft = float(acfo["object_fft"]["median_s"])
    acfo_apply = float(acfo["prepared_apply"]["median_s"])
    nufft_setup = float(nufft["plan_setup"]["median_s"])
    nufft_execute = float(nufft["execute"]["median_s"])
    rows = []
    for count in (1, 2, 4, 8, 16, 32, 64):
        acfo_total = acfo_setup + object_fft + count * acfo_apply
        nufft_total = nufft_setup + count * nufft_execute
        rows.append(
            {
                "repeat_count": count,
                "acfo_total_s": acfo_total,
                "finufft_total_s": nufft_total,
                "finufft_over_acfo": nufft_total / acfo_total,
                "acfo_setup_fraction": (acfo_setup + object_fft) / acfo_total,
                "finufft_setup_fraction": nufft_setup / nufft_total,
            }
        )
    return rows


def prepare_raw_finufft_plan(
    coords: np.ndarray,
    targets: np.ndarray,
    *,
    eps: float = 1e-9,
):
    plan = finufft.Plan(
        3,
        3,
        n_trans=1,
        eps=eps,
        isign=1,
        dtype="complex128",
        nthreads=1,
    )
    plan.setpts(
        np.ascontiguousarray(coords[:, 0]),
        np.ascontiguousarray(coords[:, 1]),
        np.ascontiguousarray(coords[:, 2]),
        np.ascontiguousarray(targets[:, 0]),
        np.ascontiguousarray(targets[:, 1]),
        np.ascontiguousarray(targets[:, 2]),
    )
    return plan


def direct_cartesian_targets(
    coords: np.ndarray,
    source_weights: np.ndarray,
    targets: np.ndarray,
    *,
    target_block: int = 16,
) -> np.ndarray:
    output = np.empty(targets.shape[0], dtype=np.complex128)
    for start in range(0, targets.shape[0], target_block):
        stop = min(start + target_block, targets.shape[0])
        phase = coords @ targets[start:stop].T
        output[start:stop] = source_weights @ np.exp(1j * phase)
    return output


def benchmark_sparse_target_control(
    *,
    n_u: int = 64,
    n_phi: int = 128,
    n_r: int = 32,
    n_z: int = 32,
    finufft_eps: float = 1e-9,
) -> list[dict[str, object]]:
    """Measure low-reuse random target subsets where full-grid ACFO does extra work."""

    histogram = make_dense_benchmark_object(n_phi, n_r=n_r, n_z=n_z)
    coords, _ = binned_structure_grid(histogram)
    source_weights = np.asarray(histogram.hist).ravel()
    manifold = matched_curvature_family(
        12.0,
        active_radius_max(histogram),
        n_u=n_u,
    )["spline"]
    all_targets = manifold.target_nodes(histogram.beta_centers).reshape(-1, 3)
    rng = np.random.default_rng(20260711)

    acfo_setup = timing(
        lambda: PreparedAxisymmetricOperator(histogram, manifold),
        repeats=3,
    )
    operator = PreparedAxisymmetricOperator(histogram, manifold)
    object_prepare = timing(
        lambda: operator.prepare_object(histogram.hist),
        repeats=5,
        warmup=1,
    )
    object_fourier = operator.prepare_object(histogram.hist)
    acfo_apply = timing(
        lambda: operator.apply_prepared_object(object_fourier),
        repeats=5,
        warmup=1,
    )
    acfo_full = operator.apply_prepared_object(object_fourier).reshape(-1)
    acfo_cold = (
        float(acfo_setup["median_s"])
        + float(object_prepare["median_s"])
        + float(acfo_apply["median_s"])
    )

    rows = []
    for target_count in (16, 32, 64, 256, 1024, 4096, all_targets.shape[0]):
        if target_count == all_targets.shape[0]:
            indices = np.arange(all_targets.shape[0])
        else:
            indices = np.sort(
                rng.choice(all_targets.shape[0], size=target_count, replace=False)
            )
        targets = np.ascontiguousarray(all_targets[indices])
        reference_subset = acfo_full[indices]
        plan_setup = timing(
            lambda: prepare_raw_finufft_plan(coords, targets, eps=finufft_eps),
            repeats=2,
        )
        plan = prepare_raw_finufft_plan(coords, targets, eps=finufft_eps)
        execute = timing(
            lambda: plan.execute(np.ascontiguousarray(source_weights)),
            repeats=5,
            warmup=1,
        )
        nufft_output = np.asarray(plan.execute(np.ascontiguousarray(source_weights)))
        nufft_cold = float(plan_setup["median_s"]) + float(execute["median_s"])
        interactions = int(coords.shape[0] * target_count)

        direct = None
        if interactions <= 5_000_000:
            direct_time = timing(
                lambda: direct_cartesian_targets(coords, source_weights, targets),
                repeats=2,
            )
            direct_output = direct_cartesian_targets(coords, source_weights, targets)
            direct = {
                "time": direct_time,
                "relative_l2_vs_acfo": relative_l2(direct_output, reference_subset),
            }

        methods = {
            "acfo_full_grid": float(acfo_apply["median_s"]),
            "finufft_selected": float(execute["median_s"]),
        }
        if direct is not None:
            methods["direct_selected"] = float(direct["time"]["median_s"])
        rows.append(
            {
                "selected_targets": target_count,
                "full_grid_targets": int(all_targets.shape[0]),
                "selection_fraction": target_count / all_targets.shape[0],
                "sources": int(coords.shape[0]),
                "interactions": interactions,
                "acfo": {
                    "geometry_setup": acfo_setup,
                    "object_fft": object_prepare,
                    "full_grid_apply": acfo_apply,
                    "cold_total_s": acfo_cold,
                },
                "finufft": {
                    "selected_plan_setup": plan_setup,
                    "selected_execute": execute,
                    "cold_total_s": nufft_cold,
                    "relative_l2_vs_acfo": relative_l2(nufft_output, reference_subset),
                },
                "direct": direct,
                "warm_fastest_method": min(methods, key=methods.get),
                "warm_method_times_s": methods,
                "finufft_execute_over_acfo_full_apply": float(execute["median_s"])
                / float(acfo_apply["median_s"]),
                "finufft_cold_over_acfo_cold": nufft_cold / acfo_cold,
            }
        )
    return rows


def benchmark_sparse_source_target_control(
    *,
    n_u: int = 64,
    n_phi: int = 128,
    finufft_eps: float = 1e-9,
) -> list[dict[str, object]]:
    """Locate the direct/structured crossover for a genuinely sparse object."""

    histogram = make_validation_object(n_phi=n_phi)
    coords, _, source_weights = binned_structure_sources(histogram)
    manifold = matched_curvature_family(
        12.0,
        active_radius_max(histogram),
        n_u=n_u,
    )["spline"]
    all_targets = manifold.target_nodes(histogram.beta_centers).reshape(-1, 3)
    rng = np.random.default_rng(20260712)

    acfo_setup = timing(
        lambda: PreparedAxisymmetricOperator(histogram, manifold),
        repeats=3,
    )
    operator = PreparedAxisymmetricOperator(histogram, manifold)
    object_fourier = operator.prepare_object(histogram.hist)
    acfo_apply = timing(
        lambda: operator.apply_prepared_object(object_fourier),
        repeats=7,
        warmup=1,
    )
    acfo_full = operator.apply_prepared_object(object_fourier).reshape(-1)

    rows = []
    for target_count in (
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
        all_targets.shape[0],
    ):
        if target_count == all_targets.shape[0]:
            indices = np.arange(all_targets.shape[0])
        else:
            indices = np.sort(
                rng.choice(all_targets.shape[0], size=target_count, replace=False)
            )
        targets = np.ascontiguousarray(all_targets[indices])
        reference_subset = acfo_full[indices]
        plan_setup = timing(
            lambda: prepare_raw_finufft_plan(coords, targets, eps=finufft_eps),
            repeats=2,
        )
        plan = prepare_raw_finufft_plan(coords, targets, eps=finufft_eps)
        execute = timing(
            lambda: plan.execute(np.ascontiguousarray(source_weights)),
            repeats=5,
            warmup=1,
        )
        nufft_output = np.asarray(plan.execute(np.ascontiguousarray(source_weights)))
        direct_time = timing(
            lambda: direct_cartesian_targets(coords, source_weights, targets),
            repeats=5,
            warmup=1,
        )
        direct_output = direct_cartesian_targets(coords, source_weights, targets)
        methods = {
            "acfo_full_grid": float(acfo_apply["median_s"]),
            "finufft_selected": float(execute["median_s"]),
            "direct_selected": float(direct_time["median_s"]),
        }
        rows.append(
            {
                "selected_targets": target_count,
                "full_grid_targets": int(all_targets.shape[0]),
                "selection_fraction": target_count / all_targets.shape[0],
                "sources": int(coords.shape[0]),
                "acfo_full_grid_apply": acfo_apply,
                "finufft_selected_plan_setup": plan_setup,
                "finufft_selected_execute": execute,
                "direct_selected": direct_time,
                "finufft_relative_l2_vs_acfo": relative_l2(nufft_output, reference_subset),
                "direct_relative_l2_vs_acfo": relative_l2(direct_output, reference_subset),
                "warm_fastest_method": min(methods, key=methods.get),
                "warm_method_times_s": methods,
            }
        )
    return rows


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# ACFO stage-6 runtime crossover and amortization benchmark",
        "",
        "All methods use the same dense cylindrical bin-center sources and the same arbitrary axisymmetric target nodes. Timings are single-thread CPU medians on the recorded local environment.",
        "",
        "## N_u sweep at N_phi=128",
        "",
        "| N_u | sources | targets | ACFO cold ms | ACFO apply ms | FINUFFT cold ms | FINUFFT exec ms | FINUFFT/ACFO warm | direct ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["n_u_sweep"]:
        direct = "-" if row["direct"] is None else f"{1e3 * row['direct']['time']['median_s']:.3f}"
        lines.append(
            f"| {row['n_u']} | {row['sources']} | {row['targets']} | "
            f"{1e3 * row['acfo']['cold_total_s']:.3f} | "
            f"{1e3 * row['acfo']['prepared_apply']['median_s']:.3f} | "
            f"{1e3 * row['finufft']['cold_total_s']:.3f} | "
            f"{1e3 * row['finufft']['execute']['median_s']:.3f} | "
            f"{row['speedups']['finufft_execute_over_acfo_apply']:.2f}x | {direct} |"
        )
    lines.extend(
        [
            "",
            "## N_phi sweep at N_u=64",
            "",
            "| N_phi | sources | targets | ACFO cold ms | ACFO apply ms | FINUFFT cold ms | FINUFFT exec ms | FINUFFT/ACFO warm | direct ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["n_phi_sweep"]:
        direct = "-" if row["direct"] is None else f"{1e3 * row['direct']['time']['median_s']:.3f}"
        lines.append(
            f"| {row['n_phi']} | {row['sources']} | {row['targets']} | "
            f"{1e3 * row['acfo']['cold_total_s']:.3f} | "
            f"{1e3 * row['acfo']['prepared_apply']['median_s']:.3f} | "
            f"{1e3 * row['finufft']['cold_total_s']:.3f} | "
            f"{1e3 * row['finufft']['execute']['median_s']:.3f} | "
            f"{row['speedups']['finufft_execute_over_acfo_apply']:.2f}x | {direct} |"
        )
    lines.extend(
        [
            "",
            "## Curvature-count batching",
            "",
            "| M | ACFO total ms | FINUFFT total ms | FINUFFT/ACFO | ACFO ms/curve |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["curvature_count_sweep"]:
        lines.append(
            f"| {row['curvature_count']} | {1e3 * row['acfo_total_s']:.3f} | "
            f"{1e3 * row['finufft_total_s']:.3f} | {row['finufft_over_acfo_total']:.2f}x | "
            f"{1e3 * row['acfo_time_per_curvature_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Repeat-count amortization for N_u=64, N_phi=128",
            "",
            "| T | ACFO total ms | FINUFFT total ms | FINUFFT/ACFO | ACFO setup fraction |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["repeat_amortization"]:
        lines.append(
            f"| {row['repeat_count']} | {1e3 * row['acfo_total_s']:.3f} | "
            f"{1e3 * row['finufft_total_s']:.3f} | {row['finufft_over_acfo']:.2f}x | "
            f"{row['acfo_setup_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sparse/random-target control on a 131,072-source object",
            "",
            "ACFO evaluates the full 8,192-node grid; FINUFFT and direct evaluate only the selected random nodes.",
            "",
            "| selected targets | fraction | ACFO full apply ms | FINUFFT selected exec ms | FINUFFT/ACFO warm | direct selected ms | fastest warm method |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["sparse_target_control"]:
        direct = "-" if row["direct"] is None else f"{1e3 * row['direct']['time']['median_s']:.3f}"
        lines.append(
            f"| {row['selected_targets']} | {row['selection_fraction']:.5f} | "
            f"{1e3 * row['acfo']['full_grid_apply']['median_s']:.3f} | "
            f"{1e3 * row['finufft']['selected_execute']['median_s']:.3f} | "
            f"{row['finufft_execute_over_acfo_full_apply']:.2f}x | {direct} | "
            f"{row['warm_fastest_method']} |"
        )
    lines.extend(
        [
            "",
            "## Sparse-source and sparse/random-target crossover",
            "",
            "The object contains only 10 nonzero sources. ACFO still evaluates the full 8,192-node grid.",
            "",
            "| selected targets | fraction | ACFO full apply ms | FINUFFT selected ms | direct selected ms | fastest warm method |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["sparse_source_target_control"]:
        lines.append(
            f"| {row['selected_targets']} | {row['selection_fraction']:.5f} | "
            f"{1e3 * row['acfo_full_grid_apply']['median_s']:.3f} | "
            f"{1e3 * row['finufft_selected_execute']['median_s']:.3f} | "
            f"{1e3 * row['direct_selected']['median_s']:.3f} | "
            f"{row['warm_fastest_method']} |"
        )
    lines.extend(
        [
            "",
            f"Accuracy pass: **{payload['passed']}**",
            "",
            "Direct timings are omitted when the declared source-target interaction limit is exceeded. Speedups are local measured-range results, not universal complexity claims.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\benchmark_axisymmetric_crossover.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "acfo_stage6_crossover.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage6_crossover.md",
    )
    args = parser.parse_args()

    cache: dict[tuple[int, int], dict[str, object]] = {}

    def case(n_u: int, n_phi: int) -> dict[str, object]:
        key = (n_u, n_phi)
        if key not in cache:
            cache[key] = benchmark_case(n_u, n_phi)
        return cache[key]

    n_u_sweep = [case(n_u, 128) for n_u in (16, 32, 64, 128, 256)]
    n_phi_sweep = [case(64, n_phi) for n_phi in (64, 96, 128, 192, 256)]
    curvature_count_sweep = [
        benchmark_curvature_count(count)
        for count in (1, 2, 4, 8)
    ]
    repeat_case = case(64, 128)
    repeats = repeat_amortization(repeat_case)
    sparse_target_control = benchmark_sparse_target_control()
    sparse_source_target_control = benchmark_sparse_source_target_control()

    accuracy_rows = list(cache.values())
    passed = all(
        row["finufft"]["relative_l2_vs_acfo"] <= 5e-8
        and (
            row["direct"] is None
            or row["direct"]["relative_l2_vs_acfo"] <= 1e-12
        )
        for row in accuracy_rows
    )
    passed = passed and all(
        row["finufft"]["relative_l2_vs_acfo"] <= 5e-8
        and (
            row["direct"] is None
            or row["direct"]["relative_l2_vs_acfo"] <= 1e-12
        )
        for row in sparse_target_control
    )
    passed = passed and all(
        row["finufft_relative_l2_vs_acfo"] <= 5e-8
        and row["direct_relative_l2_vs_acfo"] <= 1e-12
        for row in sparse_source_target_control
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage6-crossover-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "scope": "single-thread CPU measured-range benchmark",
        "method_contract": {
            "sources": "same dense cylindrical bin-center coefficients",
            "targets": "same arbitrary axisymmetric q nodes",
            "acfo_components": "geometry setup + object FFT + prepared apply",
            "finufft_components": "type-3 plan setup + execute",
            "direct_limit_interactions": 20_000_000,
            "finufft_eps": 1e-9,
            "finufft_nthreads": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "finufft": finufft.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "n_u_sweep": n_u_sweep,
        "n_phi_sweep": n_phi_sweep,
        "curvature_count_sweep": curvature_count_sweep,
        "repeat_amortization": repeats,
        "sparse_target_control": sparse_target_control,
        "sparse_source_target_control": sparse_source_target_control,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
