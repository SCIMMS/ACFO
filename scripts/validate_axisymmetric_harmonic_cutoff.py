from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    estimate_bessel_cutoff,
)
try:
    from scripts.validate_axisymmetric_manifold_discrete import make_validation_object  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from validate_axisymmetric_manifold_discrete import make_validation_object  # type: ignore[no-redef]  # noqa: E402


def active_radius_max(histogram) -> float:
    active_r = np.flatnonzero(np.any(np.asarray(histogram.hist) != 0.0, axis=(0, 2, 3)))
    if not active_r.size:
        raise ValueError("histogram must contain an active radial bin")
    return float(np.max(np.asarray(histogram.r_centers)[active_r]))


def matched_curvature_family(
    x_product: float,
    r_active: float,
    *,
    n_u: int = 24,
) -> dict[str, AxisymmetricManifold]:
    """Create distinct curves with identical ``Q_perp,max * R_active``."""

    scale = float(x_product) / float(r_active)
    u = np.linspace(0.0, 1.0, n_u)
    angle = 0.5 * np.pi * u
    spline_u = np.array([0.0, 0.18, 0.42, 0.66, 0.82, 1.0])
    spline_perp = CubicSpline(
        spline_u,
        np.array([0.0, 0.16, 0.58, 0.48, 0.81, 1.0]),
        bc_type="natural",
    )(u)
    spline_z = CubicSpline(
        spline_u,
        np.array([0.0, -0.08, -0.34, -0.18, -0.62, -0.9]),
        bc_type="natural",
    )(u)
    spline_perp = np.maximum(spline_perp, 0.0)
    spline_perp[-1] = 1.0

    family = {
        "sphere": AxisymmetricManifold(
            u,
            scale * np.sin(angle),
            scale * (np.cos(angle) - 1.0),
            name="matched-sphere",
        ),
        "ellipsoid": AxisymmetricManifold(
            u,
            scale * np.sin(angle),
            0.55 * scale * (np.cos(angle) - 1.0),
            name="matched-ellipsoid",
        ),
        "paraboloid": AxisymmetricManifold(
            u,
            scale * u,
            -0.42 * scale * u * u,
            name="matched-paraboloid",
        ),
        "spline": AxisymmetricManifold(
            u,
            scale * spline_perp,
            scale * spline_z,
            name="matched-spline",
        ),
    }
    for manifold in family.values():
        measured = float(np.max(manifold.q_perp) * r_active)
        if not np.isclose(measured, x_product, rtol=0.0, atol=1e-12):
            raise RuntimeError("matched family did not preserve Q_perp,max * R_active")
    return family


def row_truncation_error(
    data_fft: np.ndarray,
    row_index: int,
    modes: np.ndarray,
    max_h: int,
) -> float:
    row = np.asarray(data_fft[row_index], dtype=np.complex128)
    omitted = np.abs(modes) > int(max_h)
    return float(np.sqrt(np.sum(np.abs(row[omitted]) ** 2) / np.sum(np.abs(row) ** 2)))


def required_cutoff(
    data_fft: np.ndarray,
    row_index: int,
    modes: np.ndarray,
    tolerance: float,
) -> int:
    for max_h in range(int(np.max(np.abs(modes))) + 1):
        if row_truncation_error(data_fft, row_index, modes, max_h) <= tolerance:
            return max_h
    raise RuntimeError("no harmonic cutoff satisfied the requested tolerance")


def angular_interpolation_error(
    x_product: float,
    n_phi: int,
    *,
    beta: float = 0.37,
    n_dense: int = 4096,
) -> float:
    """Compare the FFT trigonometric interpolant with one analytic point source."""

    sample_phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    samples = np.exp(1j * x_product * np.cos(sample_phi - beta))
    coefficients = np.fft.fft(samples) / n_phi
    modes = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
    dense_phi = 2.0 * np.pi * np.arange(n_dense) / n_dense
    interpolated = np.exp(1j * np.outer(dense_phi, modes)) @ coefficients
    reference = np.exp(1j * x_product * np.cos(dense_phi - beta))
    return float(np.linalg.norm(interpolated - reference) / np.linalg.norm(reference))


def validate_cutoff_law() -> tuple[dict[str, object], dict[str, object]]:
    histogram = make_validation_object(n_phi=128)
    r_active = active_radius_max(histogram)
    x_products = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    tolerances = [1e-6, 1e-8, 1e-10]
    records: dict[str, object] = {}
    for x_product in x_products:
        curve_records: dict[str, object] = {}
        for family, manifold in matched_curvature_family(x_product, r_active).items():
            operator = PreparedAxisymmetricOperator(histogram, manifold)
            data_fft = operator.forward_fourier(histogram.hist)
            row_index = int(np.argmax(manifold.q_perp))
            cutoff_records: dict[str, object] = {}
            for tolerance in tolerances:
                measured = required_cutoff(
                    data_fft,
                    row_index,
                    operator.angular_modes,
                    tolerance,
                )
                predicted = estimate_bessel_cutoff(x_product, tol=tolerance)
                cutoff_records[f"{tolerance:.0e}"] = {
                    "measured_h": measured,
                    "bessel_estimate_h": predicted,
                    "estimate_margin": predicted - measured,
                    "measured_error": row_truncation_error(
                        data_fft,
                        row_index,
                        operator.angular_modes,
                        measured,
                    ),
                }
            base_h = int(np.floor(x_product))
            curve_records[family] = {
                "q_perp_max_r_active": float(np.max(manifold.q_perp) * r_active),
                "cutoffs": cutoff_records,
                "error_by_margin": {
                    str(margin): row_truncation_error(
                        data_fft,
                        row_index,
                        operator.angular_modes,
                        min(base_h + margin, histogram.n_phi // 2),
                    )
                    for margin in (0, 4, 8, 12, 16)
                },
            }
        records[f"{x_product:g}"] = curve_records

    collapse: dict[str, object] = {}
    fits: dict[str, object] = {}
    transition_fits: dict[str, object] = {}
    for tolerance in tolerances:
        key = f"{tolerance:.0e}"
        median_cutoffs = []
        max_spread = 0
        for x_product in x_products:
            values = [
                records[f"{x_product:g}"][family]["cutoffs"][key]["measured_h"]
                for family in ("sphere", "ellipsoid", "paraboloid", "spline")
            ]
            spread = max(values) - min(values)
            max_spread = max(max_spread, spread)
            median_cutoffs.append(float(np.median(values)))
        slope, intercept = np.polyfit(x_products, median_cutoffs, 1)
        predicted = slope * np.asarray(x_products) + intercept
        residual = np.asarray(median_cutoffs) - predicted
        total = np.asarray(median_cutoffs) - np.mean(median_cutoffs)
        r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
        collapse[key] = {"max_family_spread_modes": int(max_spread)}
        fits[key] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": r_squared,
            "interpretation": "empirical fit over the measured x range, not a formal complexity law",
        }
        transition_design = np.column_stack(
            (np.cbrt(np.asarray(x_products)), np.ones(len(x_products)))
        )
        transition_coefficient, transition_intercept = np.linalg.lstsq(
            transition_design,
            np.asarray(median_cutoffs) - np.asarray(x_products),
            rcond=None,
        )[0]
        transition_prediction = (
            np.asarray(x_products)
            + transition_coefficient * np.cbrt(np.asarray(x_products))
            + transition_intercept
        )
        transition_residual = np.asarray(median_cutoffs) - transition_prediction
        transition_r_squared = 1.0 - float(
            np.sum(transition_residual**2) / np.sum(total**2)
        )
        transition_fits[key] = {
            "model": "H = x + a*x^(1/3) + b",
            "a": float(transition_coefficient),
            "b": float(transition_intercept),
            "r_squared": transition_r_squared,
            "interpretation": "tolerance-dependent Bessel transition margin over the measured range",
        }
    return {
        "r_active": r_active,
        "n_phi": histogram.n_phi,
        "x_products": x_products,
        "tolerances": tolerances,
        "records": records,
    }, {"collapse": collapse, "linear_fits": fits, "transition_fits": transition_fits}


def validate_aliasing() -> dict[str, object]:
    tolerance = 1e-8
    records: dict[str, object] = {}
    for x_product in (2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
        estimated_h = estimate_bessel_cutoff(x_product, tol=tolerance)
        safe_n_phi = 2 * estimated_h + 2
        minimum_n_phi = next(
            n_phi
            for n_phi in range(8, 194, 2)
            if angular_interpolation_error(x_product, n_phi) <= tolerance
        )
        undersampled_n_phi = max(8, 2 * int(np.floor(x_product)))
        records[f"{x_product:g}"] = {
            "estimated_h": estimated_h,
            "safe_n_phi": safe_n_phi,
            "safe_error": angular_interpolation_error(x_product, safe_n_phi),
            "minimum_even_n_phi_measured": minimum_n_phi,
            "undersampled_n_phi": undersampled_n_phi,
            "undersampled_error": angular_interpolation_error(x_product, undersampled_n_phi),
        }
    return {"tolerance": tolerance, "records": records}


def render_markdown(payload: dict[str, object]) -> str:
    cutoff = payload["cutoff_validation"]
    aliasing = payload["aliasing_validation"]
    summary = payload["summary"]
    lines = [
        "# ACFO stage-5 harmonic cutoff and azimuthal aliasing validation",
        "",
        "Four distinct meridional curves are matched at the same `Q_perp,max * R_active`. Required global angular cutoffs are measured from the exact output spectrum of the same complex discrete object.",
        "",
        "## Cutoff collapse at relative tolerance 1e-8",
        "",
        "| Qperp,max Ractive | sphere H | ellipsoid H | paraboloid H | spline H | Bessel estimate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for x_value in cutoff["x_products"]:
        curves = cutoff["records"][f"{x_value:g}"]
        values = [curves[name]["cutoffs"]["1e-08"]["measured_h"] for name in ("sphere", "ellipsoid", "paraboloid", "spline")]
        estimate = curves["sphere"]["cutoffs"]["1e-08"]["bessel_estimate_h"]
        lines.append(
            f"| {x_value:g} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {estimate} |"
        )
    fit = summary["transition_fits"]["1e-08"]
    lines.extend(
        [
            "",
            f"Measured-range transition fit: `H = x + {fit['a']:.3f} x^(1/3) + {fit['b']:.3f}`, R2 = `{fit['r_squared']:.4f}`. This is a tolerance-dependent empirical summary, not a formal asymptotic statement.",
            "",
            "## Analytic single-source azimuthal aliasing",
            "",
            "| x | measured minimum even Nphi | safe Nphi from cutoff | safe error | undersampled Nphi | undersampled error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for x_value, raw in aliasing["records"].items():
        lines.append(
            f"| {x_value} | {raw['minimum_even_n_phi_measured']} | {raw['safe_n_phi']} | "
            f"{raw['safe_error']:.3e} | {raw['undersampled_n_phi']} | {raw['undersampled_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_harmonic_cutoff.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage5_harmonic_cutoff.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage5_harmonic_cutoff.md",
    )
    args = parser.parse_args()

    cutoff, summary = validate_cutoff_law()
    aliasing = validate_aliasing()
    all_estimates_conservative = all(
        raw["cutoffs"][tol]["estimate_margin"] >= 0
        for curves in cutoff["records"].values()
        for raw in curves.values()
        for tol in ("1e-06", "1e-08", "1e-10")
    )
    collapse_ok = all(
        raw["max_family_spread_modes"] <= 4
        for raw in summary["collapse"].values()
    )
    fit_ok = all(
        raw["r_squared"] >= 0.995
        for raw in summary["transition_fits"].values()
    )
    alias_ok = all(
        raw["safe_error"] <= aliasing["tolerance"]
        and raw["minimum_even_n_phi_measured"] <= raw["safe_n_phi"]
        and raw["undersampled_error"] > 100.0 * aliasing["tolerance"]
        for raw in aliasing["records"].values()
    )
    passed = all_estimates_conservative and collapse_ok and fit_ok and alias_ok
    payload: dict[str, object] = {
        "schema": "acfo-stage5-harmonic-cutoff-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "acceptance": {
            "max_family_spread_modes": 4,
            "transition_model": "H = x + a*x^(1/3) + b",
            "transition_fit_r_squared_min": 0.995,
            "bessel_estimate_must_be_conservative": True,
            "safe_alias_error_max": aliasing["tolerance"],
        },
        "cutoff_validation": cutoff,
        "aliasing_validation": aliasing,
        "summary": summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
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
