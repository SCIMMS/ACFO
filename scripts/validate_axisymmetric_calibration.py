from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AnchoredBSplineCurvatureModel,
    EllipsoidCurvatureModel,
    PreparedAxisymmetricOperator,
    curvature_loss_and_gradient,
    geometry_identifiability,
)
from validate_axisymmetric_manifold_discrete import make_validation_object  # noqa: E402


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def normalized_complex_noise(
    data: np.ndarray,
    relative_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if relative_norm == 0.0:
        return np.zeros_like(data)
    noise = rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
    return noise * (relative_norm * np.linalg.norm(data) / np.linalg.norm(noise))


def finite_difference_gradient(
    template,
    model,
    parameters: np.ndarray,
    data: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, object]:
    loss, analytic = curvature_loss_and_gradient(template, model, parameters, data)
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        step = np.zeros_like(parameters)
        step[index] = epsilon
        plus = curvature_loss_and_gradient(template, model, parameters + step, data)[0]
        minus = curvature_loss_and_gradient(template, model, parameters - step, data)[0]
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)
    denominator = max(np.linalg.norm(finite_difference), np.finfo(np.float64).tiny)
    return {
        "loss": loss,
        "analytic": analytic.tolist(),
        "finite_difference": finite_difference.tolist(),
        "relative_l2": float(np.linalg.norm(analytic - finite_difference) / denominator),
        "epsilon": epsilon,
    }


def directional_geometry_check(
    template,
    model,
    parameters: np.ndarray,
    rng: np.random.Generator,
    *,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    direction = rng.normal(size=model.n_parameters)
    direction /= np.linalg.norm(direction)
    manifold = model.manifold(parameters)
    operator = PreparedAxisymmetricOperator(template, manifold)
    jacobian_perp, jacobian_z = model.geometry_jacobians(parameters)
    analytic = operator.geometry_jacobian_action(
        template.hist,
        jacobian_perp @ direction,
        jacobian_z @ direction,
    )
    plus = PreparedAxisymmetricOperator(
        template,
        model.manifold(parameters + epsilon * direction),
    ).forward(template.hist)
    minus = PreparedAxisymmetricOperator(
        template,
        model.manifold(parameters - epsilon * direction),
    ).forward(template.hist)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    return {
        "relative_l2": relative_l2(analytic, finite_difference),
        "epsilon": epsilon,
    }


def optimize_geometry(
    template,
    model,
    initial: np.ndarray,
    data: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    smoothness: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return curvature_loss_and_gradient(
            template,
            model,
            values,
            data,
            smoothness=smoothness,
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10, "maxls": 40},
    )
    estimate = np.asarray(result.x, dtype=np.float64)
    prediction = PreparedAxisymmetricOperator(template, model.manifold(estimate)).forward(
        template.hist
    )
    summary = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_objective": float(result.fun),
        "relative_data_residual": relative_l2(prediction, data),
        "estimate": estimate.tolist(),
        "smoothness": smoothness,
    }
    return estimate, summary


def local_uncertainty(
    identifiability: dict[str, object],
    data: np.ndarray,
    relative_noise: float,
) -> list[float]:
    singular_values = np.asarray(identifiability["singular_values"], dtype=np.float64)
    sigma_real = relative_noise * np.linalg.norm(data) / np.sqrt(2.0 * data.size)
    # SVD-only summary: principal-axis standard deviations in descending
    # certainty order. Parameter-axis covariance is intentionally not implied.
    return (sigma_real / singular_values).tolist()


def validate_ellipsoid(template, rng: np.random.Generator) -> dict[str, object]:
    model = EllipsoidCurvatureModel(np.linspace(0.02, 1.2, 32))
    truth = np.array([3.0, 1.4])
    initial = np.array([2.25, 2.0])
    clean = PreparedAxisymmetricOperator(template, model.manifold(truth)).forward(template.hist)
    gradient_check = finite_difference_gradient(template, model, initial, clean)
    identifiability = geometry_identifiability(template, model, truth)
    cases: dict[str, object] = {}
    for relative_noise in (0.0, 1e-3, 1e-2):
        observed = clean + normalized_complex_noise(clean, relative_noise, rng)
        estimate, summary = optimize_geometry(
            template,
            model,
            initial,
            observed,
            [(0.2, 5.0), (0.2, 5.0)],
        )
        summary["parameter_relative_l2"] = relative_l2(estimate, truth)
        summary["relative_noise_norm"] = relative_noise
        cases[f"noise_{relative_noise:.0e}"] = summary
    return {
        "truth": truth.tolist(),
        "initial": initial.tolist(),
        "initial_parameter_relative_l2": relative_l2(initial, truth),
        "objective_gradient_check": gradient_check,
        "identifiability": identifiability,
        "principal_axis_standard_deviation_at_1pct_noise": local_uncertainty(
            identifiability, clean, 1e-2
        ),
        "recoveries": cases,
    }


def validate_spline(template, rng: np.random.Generator) -> dict[str, object]:
    model = AnchoredBSplineCurvatureModel(np.linspace(0.0, 1.0, 32))
    truth_perp = np.array([0.0, 0.45, 1.9, 1.35, 2.8])
    truth_z = np.array([0.0, -0.15, -0.65, -0.25, -1.1])
    initial_perp = np.array([0.0, 0.52, 1.65, 1.50, 2.55])
    initial_z = np.array([0.0, -0.10, -0.52, -0.40, -0.92])
    truth = model.parameters_from_controls(truth_perp, truth_z)
    initial = model.parameters_from_controls(initial_perp, initial_z)
    truth_manifold = model.manifold(truth)
    clean = PreparedAxisymmetricOperator(template, truth_manifold).forward(template.hist)
    gradient_check = finite_difference_gradient(template, model, initial, clean)
    directional_check = directional_geometry_check(template, model, truth, rng)
    identifiability = geometry_identifiability(template, model, truth)
    bounds = [(0.0, 4.0)] * (model.n_control - 1) + [(-3.0, 1.0)] * (
        model.n_control - 1
    )
    cases: dict[str, object] = {}
    for relative_noise, smoothness in ((0.0, 0.0), (1e-3, 1e-8), (1e-2, 1e-7)):
        observed = clean + normalized_complex_noise(clean, relative_noise, rng)
        estimate, summary = optimize_geometry(
            template,
            model,
            initial,
            observed,
            bounds,
            smoothness=smoothness,
        )
        estimated_manifold = model.manifold(estimate)
        curve_truth = np.concatenate((truth_manifold.q_perp, truth_manifold.q_z))
        curve_estimate = np.concatenate(
            (estimated_manifold.q_perp, estimated_manifold.q_z)
        )
        summary["parameter_relative_l2"] = relative_l2(estimate, truth)
        summary["curve_relative_l2"] = relative_l2(curve_estimate, curve_truth)
        summary["relative_noise_norm"] = relative_noise
        cases[f"noise_{relative_noise:.0e}"] = summary
    initial_manifold = model.manifold(initial)
    return {
        "truth_q_perp_controls": truth_perp.tolist(),
        "truth_q_z_controls": truth_z.tolist(),
        "initial_q_perp_controls": initial_perp.tolist(),
        "initial_q_z_controls": initial_z.tolist(),
        "initial_curve_relative_l2": relative_l2(
            np.concatenate((initial_manifold.q_perp, initial_manifold.q_z)),
            np.concatenate((truth_manifold.q_perp, truth_manifold.q_z)),
        ),
        "objective_gradient_check": gradient_check,
        "directional_geometry_check": directional_check,
        "identifiability": identifiability,
        "principal_axis_standard_deviation_at_1pct_noise": local_uncertainty(
            identifiability, clean, 1e-2
        ),
        "recoveries": cases,
    }


def render_markdown(payload: dict[str, object]) -> str:
    ellipsoid = payload["ellipsoid"]
    spline = payload["anchored_bspline"]
    assert isinstance(ellipsoid, dict) and isinstance(spline, dict)
    lines = [
        "# ACFO stage-7 curvature calibration validation",
        "",
        "This stage calibrates sampling geometry from a known discrete complex phantom. The object, azimuth grid, and constant unit scattering factors are fixed; only the meridional curvature parameters are estimated.",
        "",
        "## Exact differentiation",
        "",
        f"- ellipsoid objective-gradient finite-difference relative L2: `{ellipsoid['objective_gradient_check']['relative_l2']:.3e}`",
        f"- anchored spline objective-gradient finite-difference relative L2: `{spline['objective_gradient_check']['relative_l2']:.3e}`",
        f"- spline parameter-direction geometry Jacobian relative L2: `{spline['directional_geometry_check']['relative_l2']:.3e}`",
        "",
        "## Synthetic recovery",
        "",
        "| model | relative noise | parameter relative L2 | curve relative L2 | data residual | iterations | success |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, result in (("ellipsoid", ellipsoid), ("anchored B-spline", spline)):
        for case in result["recoveries"].values():
            curve = case.get("curve_relative_l2")
            curve_text = "-" if curve is None else f"{curve:.3e}"
            lines.append(
                f"| {model_name} | {case['relative_noise_norm']:.0e} | "
                f"{case['parameter_relative_l2']:.3e} | {curve_text} | "
                f"{case['relative_data_residual']:.3e} | {case['iterations']} | "
                f"{case['success']} |"
            )
    lines.extend(
        [
            "",
            "## Local identifiability at truth",
            "",
            "| model | rank | parameters | Jacobian condition number |",
            "|---|---:|---:|---:|",
            f"| ellipsoid | {ellipsoid['identifiability']['rank']} | {ellipsoid['identifiability']['n_parameters']} | {ellipsoid['identifiability']['condition_number']:.3e} |",
            f"| anchored B-spline | {spline['identifiability']['rank']} | {spline['identifiability']['n_parameters']} | {spline['identifiability']['condition_number']:.3e} |",
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "The first spline control point is fixed at `(Q_perp, Q_z) = (0, 0)`, and `u` is fixed. This removes the endpoint and reparameterization gauges from the tested inverse problem. The uncertainty values in JSON are local linear principal-axis estimates, not global confidence intervals.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_calibration.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage7_calibration.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage7_calibration.md",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(20260711)
    template = make_validation_object()
    ellipsoid = validate_ellipsoid(template, rng)
    spline = validate_spline(template, rng)
    acceptance = {
        "gradient_relative_l2_max": 1e-6,
        "directional_geometry_relative_l2_max": 1e-6,
        "ellipsoid_noiseless_parameter_relative_l2_max": 1e-5,
        "ellipsoid_1pct_parameter_relative_l2_max": 0.1,
        "spline_noiseless_curve_relative_l2_max": 1e-4,
        "spline_1pct_curve_relative_l2_max": 0.1,
        "full_rank_required": True,
    }
    ellipsoid_recovery = ellipsoid["recoveries"]
    spline_recovery = spline["recoveries"]
    passed = bool(
        ellipsoid["objective_gradient_check"]["relative_l2"]
        <= acceptance["gradient_relative_l2_max"]
        and spline["objective_gradient_check"]["relative_l2"]
        <= acceptance["gradient_relative_l2_max"]
        and spline["directional_geometry_check"]["relative_l2"]
        <= acceptance["directional_geometry_relative_l2_max"]
        and ellipsoid_recovery["noise_0e+00"]["parameter_relative_l2"]
        <= acceptance["ellipsoid_noiseless_parameter_relative_l2_max"]
        and ellipsoid_recovery["noise_1e-02"]["parameter_relative_l2"]
        <= acceptance["ellipsoid_1pct_parameter_relative_l2_max"]
        and spline_recovery["noise_0e+00"]["curve_relative_l2"]
        <= acceptance["spline_noiseless_curve_relative_l2_max"]
        and spline_recovery["noise_1e-02"]["curve_relative_l2"]
        <= acceptance["spline_1pct_curve_relative_l2_max"]
        and ellipsoid["identifiability"]["rank"]
        == ellipsoid["identifiability"]["n_parameters"]
        and spline["identifiability"]["rank"]
        == spline["identifiability"]["n_parameters"]
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage7-curvature-calibration-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "calibration_contract": {
            "known_object": True,
            "fixed_u": True,
            "constant_unit_form_factors": True,
            "euclidean_complex_data_loss": True,
            "spline_first_control_fixed": [0.0, 0.0],
        },
        "phantom": {
            "nonzero_complex_sources": int(np.count_nonzero(template.hist)),
            "n_phi": int(template.beta_centers.size),
            "multiple_z_planes": True,
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "ellipsoid": ellipsoid,
        "anchored_bspline": spline,
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
