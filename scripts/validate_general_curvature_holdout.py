from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    BinnedStructure,
    PreparedAxisymmetricOperator,
    binned_structure_grid,
    binned_structure_sources,
    complex_error_metrics,
    direct_axisymmetric_adjoint,
    direct_axisymmetric_amplitude,
)


SOURCE_SEEDS = (2026071401, 2026071402, 2026071403, 2026071404, 2026071405)


def _data_weights(u: np.ndarray, phase: float) -> np.ndarray:
    span = float(np.ptp(u))
    scaled = (u - u[0]) / span if span > 0.0 else np.zeros_like(u)
    return 0.25 + 0.55 * scaled + 0.35 * np.sin(np.pi * scaled + phase) ** 2


def make_holdout_manifolds(n_u: int = 48) -> dict[str, AxisymmetricManifold]:
    """Return untuned manifolds, including near-axis and high-curvature controls."""

    if n_u < 12:
        raise ValueError("n_u must be at least 12")

    # The parameters differ from the original sphere/ellipsoid/paraboloid/spline
    # validation and are intentionally kept local to this holdout suite.
    q_sphere = np.linspace(0.0, 4.65, n_u)
    sphere = AxisymmetricManifold.ewald_sphere(
        q_sphere,
        wavelength=0.83,
        data_weights=_data_weights(q_sphere, 0.1),
        name="holdout-shifted-ewald-sphere",
    )

    u_ellipsoid = np.linspace(0.0, 1.38, n_u)
    ellipsoid = AxisymmetricManifold.from_callback(
        u_ellipsoid,
        lambda u: (4.25 * np.sin(u), 2.35 * (np.cos(u) - 1.0) - 0.07 * u),
        data_weights=_data_weights(u_ellipsoid, 0.4),
        name="holdout-ellipsoid",
    )

    u_paraboloid = np.linspace(0.0, 4.15, n_u)
    paraboloid = AxisymmetricManifold.from_callback(
        u_paraboloid,
        lambda u: (u, 0.12 * u - 0.21 * u * u + 0.006 * u**3),
        data_weights=_data_weights(u_paraboloid, 0.8),
        name="holdout-open-cubic-paraboloid",
    )

    u_spline = np.linspace(0.0, 1.0, n_u)
    spline_knots = np.array([0.0, 0.11, 0.31, 0.52, 0.71, 0.89, 1.0])
    q_perp_spline = CubicSpline(
        spline_knots,
        np.array([0.16, 0.72, 1.94, 1.28, 3.38, 2.83, 4.42]),
        bc_type="natural",
    )(u_spline)
    q_z_spline = CubicSpline(
        spline_knots,
        np.array([0.08, -0.22, -0.95, -0.36, -1.72, -1.18, -2.55]),
        bc_type="natural",
    )(u_spline)
    if np.any(q_perp_spline < 0.0):
        raise RuntimeError("holdout spline produced negative q_perp")
    spline = AxisymmetricManifold(
        u_spline,
        q_perp_spline,
        q_z_spline,
        data_weights=_data_weights(u_spline, 1.1),
        name="holdout-tabulated-spline",
    )

    u_strong = np.linspace(0.0, 1.0, n_u)
    q_perp_strong = 0.12 + 4.25 * u_strong + 0.34 * np.sin(8.0 * np.pi * u_strong)
    q_z_strong = -2.8 * u_strong**2 + 0.52 * np.sin(7.0 * np.pi * u_strong + 0.2)
    if np.any(q_perp_strong < 0.0):
        raise RuntimeError("strong-curvature control produced negative q_perp")
    strong_curvature = AxisymmetricManifold(
        u_strong,
        q_perp_strong,
        q_z_strong,
        data_weights=_data_weights(u_strong, 1.5),
        name="strong-curvature-oscillatory-holdout",
    )

    u_axis = np.linspace(0.0, 1.0, n_u)
    # The first point is exactly on axis and the next points grow cubically,
    # so the factorized operator is exercised at and immediately around q_perp=0.
    q_perp_axis = 4.1 * u_axis**3
    q_z_axis = -0.15 * u_axis - 2.15 * u_axis**2 + 0.18 * np.sin(3.0 * np.pi * u_axis)
    near_axis = AxisymmetricManifold(
        u_axis,
        q_perp_axis,
        q_z_axis,
        data_weights=_data_weights(u_axis, 1.9),
        name="near-axis-cubic-departure-holdout",
    )

    return {
        "shifted_sphere": sphere,
        "ellipsoid_holdout": ellipsoid,
        "open_cubic_paraboloid": paraboloid,
        "tabulated_spline_holdout": spline,
        "strong_curvature": strong_curvature,
        "near_axis": near_axis,
    }


def make_holdout_object(seed: int, n_phi: int = 64, nonzero_sources: int = 36) -> BinnedStructure:
    """Build one deterministic sparse complex cylindrical source realization."""

    if n_phi < 16:
        raise ValueError("n_phi must be at least 16")
    rng = np.random.default_rng(seed)
    n_r, n_z, n_elements = 4, 4, 2
    r_edges = np.array([0.0, 0.31, 0.86, 1.58, 2.55])
    z_edges = np.array([-1.9, -0.74, 0.05, 0.91, 1.83])
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    shape = (n_elements, n_r, n_z, n_phi)
    histogram = np.zeros(shape, dtype=np.complex128)
    flat_indices = rng.choice(np.prod(shape), size=nonzero_sources, replace=False)
    amplitudes = rng.normal(size=nonzero_sources) + 1j * rng.normal(size=nonzero_sources)
    amplitudes *= 0.4 + rng.random(nonzero_sources)
    histogram.ravel()[flat_indices] = amplitudes

    return BinnedStructure(
        hist=histogram,
        r_centers=0.5 * (r_edges[:-1] + r_edges[1:]),
        z_centers=0.5 * (z_edges[:-1] + z_edges[1:]),
        beta_centers=0.5 * (beta_edges[:-1] + beta_edges[1:]),
        elements=("A", "B"),
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )


def holdout_form_factors() -> dict[str, object]:
    return {
        "A": lambda q: 1.0 + 0.035 * q + 0.025j,
        "B": lambda q: 0.78 - 0.022 * q + 0.09j,
    }


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(np.asarray(reference).ravel())
    if denominator == 0.0:
        raise ValueError("reference must have nonzero norm")
    return float(np.linalg.norm((np.asarray(actual) - np.asarray(reference)).ravel()) / denominator)


def _curve_diagnostics(manifold: AxisymmetricManifold) -> dict[str, float | int]:
    u = np.asarray(manifold.u)
    q_perp = np.asarray(manifold.q_perp)
    q_z = np.asarray(manifold.q_z)
    edge_order = 2 if u.size >= 3 else 1
    dr = np.gradient(q_perp, u, edge_order=edge_order)
    dz = np.gradient(q_z, u, edge_order=edge_order)
    ddr = np.gradient(dr, u, edge_order=edge_order)
    ddz = np.gradient(dz, u, edge_order=edge_order)
    speed = np.hypot(dr, dz)
    curvature = np.abs(dr * ddz - dz * ddr) / np.maximum(speed**3, 1e-15)
    tangent_angle = np.unwrap(np.arctan2(dz, dr))
    return {
        "q_perp_min": float(np.min(q_perp)),
        "q_perp_max": float(np.max(q_perp)),
        "q_z_min": float(np.min(q_z)),
        "q_z_max": float(np.max(q_z)),
        "near_axis_samples_qperp_le_1e-3": int(np.count_nonzero(q_perp <= 1e-3)),
        "max_discrete_meridional_curvature": float(np.max(curvature)),
        "total_absolute_tangent_turn_rad": float(np.sum(np.abs(np.diff(tangent_angle)))),
    }


def validate_holdouts(
    *,
    seeds: tuple[int, ...] = SOURCE_SEEDS,
    n_u: int = 48,
    n_phi: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifolds = make_holdout_manifolds(n_u=n_u)
    factors = holdout_form_factors()
    cases: dict[str, Any] = {}
    worst = {
        "complex128_forward_relative_l2": 0.0,
        "complex128_weighted_adjoint_relative_l2": 0.0,
        "complex128_weighted_dot_error": 0.0,
        "complex64_forward_relative_l2": 0.0,
        "complex64_weighted_adjoint_relative_l2": 0.0,
        "complex64_weighted_dot_error": 0.0,
    }

    for seed in seeds:
        template = make_holdout_object(seed, n_phi=n_phi)
        source_coords, source_elements, source_weights = binned_structure_sources(template)
        grid_coords, grid_elements = binned_structure_grid(template)
        rng = np.random.default_rng(seed ^ 0x5A17)
        seed_results: dict[str, Any] = {}

        for family, manifold in manifolds.items():
            phi = np.asarray(template.beta_centers)
            direct_forward = direct_axisymmetric_amplitude(
                source_coords,
                manifold,
                phi,
                elements=source_elements,
                form_factors=factors,
                source_weights=source_weights,
            )
            data_values = rng.normal(size=direct_forward.shape) + 1j * rng.normal(
                size=direct_forward.shape
            )
            direct_weighted_adjoint = direct_axisymmetric_adjoint(
                grid_coords,
                manifold,
                phi,
                data_values,
                elements=grid_elements,
                form_factors=factors,
                data_weights=manifold.resolved_data_weights,
            ).reshape(np.asarray(template.hist).shape)

            dtype_results: dict[str, Any] = {}
            for dtype in (np.complex128, np.complex64):
                dtype_name = np.dtype(dtype).name
                operator = PreparedAxisymmetricOperator(
                    template,
                    manifold,
                    form_factors=factors,
                    complex_dtype=dtype,
                )
                object_values = np.asarray(template.hist, dtype=dtype)
                prepared_forward = operator.forward(object_values)
                prepared_adjoint = operator.adjoint_weighted(data_values)
                forward_metrics = complex_error_metrics(prepared_forward, direct_forward).to_dict()
                adjoint_relative_l2 = _relative_l2(
                    prepared_adjoint,
                    direct_weighted_adjoint,
                )
                dot_error = operator.adjoint_test(object_values, data_values, weighted=True)
                dtype_results[dtype_name] = {
                    "forward_complex_error": forward_metrics,
                    "weighted_adjoint_relative_l2": adjoint_relative_l2,
                    "weighted_dot_error": dot_error,
                }
                prefix = f"{dtype_name}_"
                worst[prefix + "forward_relative_l2"] = max(
                    worst[prefix + "forward_relative_l2"],
                    float(forward_metrics["relative_l2"]),
                )
                worst[prefix + "weighted_adjoint_relative_l2"] = max(
                    worst[prefix + "weighted_adjoint_relative_l2"],
                    adjoint_relative_l2,
                )
                worst[prefix + "weighted_dot_error"] = max(
                    worst[prefix + "weighted_dot_error"],
                    dot_error,
                )

            seed_results[family] = dtype_results
        cases[str(seed)] = seed_results

    curve_diagnostics = {
        family: _curve_diagnostics(manifold) for family, manifold in manifolds.items()
    }
    return cases, {"worst_case": worst, "curve_diagnostics": curve_diagnostics}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    worst = summary["worst_case"]
    diagnostics = summary["curve_diagnostics"]
    acceptance = payload["acceptance"]
    lines = [
        "# General-curvature R2 publication holdout validation",
        "",
        "이 검증은 기존 4개 예제와 분리된 holdout이다. 순환 FFT/조화 전개를 쓰지 않는 Cartesian complex exponent sum을 forward 및 weighted adjoint reference로 사용한다.",
        "",
        f"- 곡률군: `{payload['problem']['curve_families']}`",
        f"- 독립 source realization: `{payload['problem']['source_realizations']}`",
        f"- 총 dtype별 case: `{payload['problem']['curve_families'] * payload['problem']['source_realizations']}`",
        f"- azimuth samples: `{payload['problem']['n_phi']}`; meridional samples: `{payload['problem']['n_u']}`",
        "",
        "## Worst-case errors",
        "",
        "| dtype | forward relative L2 | weighted adjoint relative L2 | weighted dot error | gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for dtype in ("complex128", "complex64"):
        lines.append(
            f"| {dtype} | {worst[dtype + '_forward_relative_l2']:.3e} | "
            f"{worst[dtype + '_weighted_adjoint_relative_l2']:.3e} | "
            f"{worst[dtype + '_weighted_dot_error']:.3e} | "
            f"{acceptance[dtype + '_relative_error_max']:.1e} |"
        )

    lines.extend(
        [
            "",
            "## Curve stress diagnostics",
            "",
            "| family | q_perp range | q_z range | near-axis samples | max discrete curvature | total tangent turn (rad) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family, raw in diagnostics.items():
        lines.append(
            f"| {family} | {raw['q_perp_min']:.3g} to {raw['q_perp_max']:.3g} | "
            f"{raw['q_z_min']:.3g} to {raw['q_z_max']:.3g} | "
            f"{raw['near_axis_samples_qperp_le_1e-3']} | "
            f"{raw['max_discrete_meridional_curvature']:.3g} | "
            f"{raw['total_absolute_tangent_turn_rad']:.3g} |"
        )
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "해석 범위: 이 결과는 임의의 유한 axisymmetric meridional sampling curve에 대한 discrete prepared forward/adjoint의 정확성을 검증한다. 특정 곡면이 물리적 dispersion surface라는 사실은 별도의 R3 물리 reference가 담당한다.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_general_curvature_holdout.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-u", type=int, default=48)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "general_curvature_r2_holdout.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "general_curvature_r2_holdout_ko.md",
    )
    args = parser.parse_args()

    cases, summary = validate_holdouts(n_u=args.n_u, n_phi=args.n_phi)
    acceptance = {
        "complex128_relative_error_max": 1e-11,
        "complex64_relative_error_max": 3e-6,
    }
    worst = summary["worst_case"]
    passed = all(
        worst[f"{dtype}_{metric}"] <= acceptance[f"{dtype}_relative_error_max"]
        for dtype in ("complex128", "complex64")
        for metric in (
            "forward_relative_l2",
            "weighted_adjoint_relative_l2",
            "weighted_dot_error",
        )
    )
    payload: dict[str, Any] = {
        "schema": "general-curvature-r2-publication-holdout-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "problem": {
            "curve_families": 6,
            "source_realizations": len(SOURCE_SEEDS),
            "source_seeds": list(SOURCE_SEEDS),
            "nonzero_complex_sources_per_realization": 36,
            "n_u": args.n_u,
            "n_phi": args.n_phi,
            "elements": ["A", "B"],
            "complex_q_dependent_form_factors": True,
        },
        "reference": {
            "forward": "independent Cartesian complex exponent sum over nonzero bin centers",
            "adjoint": "independent Cartesian conjugate exponent sum over the full cylindrical grid",
            "shared_factorization_with_acfo": False,
            "per_case_fitting": False,
        },
        "inner_products": {
            "object": "Euclidean over discrete cylindrical coefficients",
            "data": "positive radial data_weights[u], broadcast over phi",
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "summary": summary,
        "cases": cases,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": passed, **summary}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
