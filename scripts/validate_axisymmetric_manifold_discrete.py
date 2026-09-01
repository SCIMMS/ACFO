from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    BinnedStructure,
    binned_structure_sources,
    complex_error_metrics,
    direct_axisymmetric_amplitude,
    prepare_axisymmetric_plan,
)


def make_validation_object(n_phi: int = 128) -> BinnedStructure:
    r_edges = np.linspace(0.0, 2.4, 7)
    z_edges = np.linspace(-1.8, 1.8, 7)
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    histogram = np.zeros((1, r_centers.size, z_centers.size, n_phi), dtype=np.complex128)

    locations = [
        (0, 0, 3),
        (1, 4, 17),
        (2, 1, 29),
        (3, 5, 43),
        (4, 2, 58),
        (5, 3, 71),
        (1, 0, 86),
        (2, 5, 99),
        (4, 4, 111),
        (5, 1, 123),
    ]
    weights = np.array(
        [
            1.0 + 0.2j,
            -0.7 + 0.4j,
            0.5 - 0.9j,
            -0.35 - 0.25j,
            0.8 + 0.0j,
            -0.4 + 0.7j,
            0.2 - 0.5j,
            0.65 + 0.3j,
            -0.55 + 0.1j,
            0.3 + 0.6j,
        ],
        dtype=np.complex128,
    )
    for (r_index, z_index, beta_index), weight in zip(locations, weights, strict=True):
        histogram[0, r_index, z_index, beta_index] = weight

    return BinnedStructure(
        hist=histogram,
        r_centers=r_centers,
        z_centers=z_centers,
        beta_centers=beta_centers,
        elements=("X",),
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )


def make_curvature_family(n_u: int = 64) -> dict[str, AxisymmetricManifold]:
    sphere_q = np.linspace(0.08, 4.0, n_u)
    ellipsoid_u = np.linspace(0.02, 1.25, n_u)
    paraboloid_u = np.linspace(0.02, 3.8, n_u)
    spline_u = np.linspace(0.0, 1.0, n_u)
    control_u = np.array([0.0, 0.18, 0.42, 0.68, 0.84, 1.0])
    q_perp_spline = CubicSpline(
        control_u,
        np.array([0.08, 0.9, 2.1, 1.45, 3.2, 3.7]),
        bc_type="natural",
    )(spline_u)
    q_z_spline = CubicSpline(
        control_u,
        np.array([0.0, -0.35, -0.08, -1.1, -0.55, -1.45]),
        bc_type="natural",
    )(spline_u)
    if np.any(q_perp_spline < 0.0):
        raise RuntimeError("spline control points produced negative Q_perp")

    return {
        "sphere": AxisymmetricManifold.ewald_sphere(
            sphere_q,
            wavelength=1.0,
            name="elastic-ewald-sphere",
        ),
        "ellipsoid": AxisymmetricManifold.from_callback(
            ellipsoid_u,
            lambda u: (3.7 * np.sin(u), 1.8 * (np.cos(u) - 1.0)),
            name="ellipsoid",
        ),
        "paraboloid": AxisymmetricManifold.from_callback(
            paraboloid_u,
            lambda u: (u, 0.08 * u - 0.16 * u * u),
            name="open-paraboloid",
        ),
        "spline": AxisymmetricManifold(
            spline_u,
            q_perp_spline,
            q_z_spline,
            name="tabulated-cubic-spline",
        ),
    }


def validate_family(
    binned: BinnedStructure,
    manifolds: dict[str, AxisymmetricManifold],
) -> dict[str, dict[str, object]]:
    coords, elements, weights = binned_structure_sources(binned)
    phi = binned.beta_centers
    results: dict[str, dict[str, object]] = {}
    for family, manifold in manifolds.items():
        reference = direct_axisymmetric_amplitude(
            coords,
            manifold,
            phi,
            elements=elements,
            source_weights=weights,
        )
        actual = prepare_axisymmetric_plan(
            binned,
            manifold,
            circular_backend="numpy",
            complex_dtype=np.complex128,
        ).circular_fft()
        metrics = complex_error_metrics(actual, reference)

        flat_manifold = AxisymmetricManifold(
            manifold.u,
            manifold.q_perp,
            np.zeros(manifold.n_u),
            name=f"{family}-flat-qz-control",
        )
        flat_reference = direct_axisymmetric_amplitude(
            coords,
            flat_manifold,
            phi,
            elements=elements,
            source_weights=weights,
        )
        flat_qz_relative_l2 = float(
            np.linalg.norm((flat_reference - reference).ravel())
            / np.linalg.norm(reference.ravel())
        )
        results[family] = {
            "n_u": manifold.n_u,
            "n_phi": int(phi.size),
            "q_perp_max": float(np.max(manifold.q_perp)),
            "q_z_min": float(np.min(manifold.q_z)),
            "q_z_max": float(np.max(manifold.q_z)),
            "complex_error": metrics.to_dict(),
            "flat_qz_control_relative_l2": flat_qz_relative_l2,
        }
    return results


def render_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, dict)
    lines = [
        "# ACFO stage-2 discrete arbitrary-curvature validation",
        "",
        "The prepared circular operator and the reference use the same complex-valued bin-center point sources. The reference is an independent Cartesian exponent sum and uses no angular FFT, Bessel function, or harmonic factorization.",
        "",
        "| family | relative L2 | relative Linf | phase RMS (rad) | flat-Qz control L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, raw in results.items():
        assert isinstance(raw, dict)
        error = raw["complex_error"]
        assert isinstance(error, dict)
        lines.append(
            f"| {family} | {error['relative_l2']:.3e} | {error['relative_linf']:.3e} | "
            f"{error['phase_rms_rad']:.3e} | {raw['flat_qz_control_relative_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "Acceptance thresholds:",
            "",
            f"- complex relative L2 <= `{payload['acceptance']['relative_l2_max']:.1e}`",
            f"- complex relative Linf <= `{payload['acceptance']['relative_linf_max']:.1e}`",
            f"- phase RMS <= `{payload['acceptance']['phase_rms_rad_max']:.1e}` rad",
            f"- flat-Qz control relative L2 >= `{payload['acceptance']['flat_qz_control_min']:.1e}`",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_manifold_discrete.py",
            "```",
            "",
            "This stage isolates discrete operator correctness. Continuous Gaussian discretization error, NUFFT comparison, and adjoint validation are not included here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "acfo_stage2_discrete_validation.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage2_discrete_validation.md",
    )
    args = parser.parse_args()

    acceptance = {
        "relative_l2_max": 1e-12,
        "relative_linf_max": 1e-12,
        "phase_rms_rad_max": 1e-12,
        "flat_qz_control_min": 1e-2,
    }
    binned = make_validation_object()
    results = validate_family(binned, make_curvature_family())
    passed = all(
        raw["complex_error"]["relative_l2"] <= acceptance["relative_l2_max"]
        and raw["complex_error"]["relative_linf"] <= acceptance["relative_linf_max"]
        and raw["complex_error"]["phase_rms_rad"] <= acceptance["phase_rms_rad_max"]
        and raw["flat_qz_control_relative_l2"] >= acceptance["flat_qz_control_min"]
        for raw in results.values()
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage2-discrete-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "source_model": {
            "representation": "nonzero BinnedStructure bin centers",
            "nonzero_sources": int(np.count_nonzero(binned.hist)),
            "complex_weights": True,
            "multiple_z_planes": True,
        },
        "reference": "direct Cartesian exponent sum",
        "backend": "PreparedCakePlan circular FFT, NumPy complex128",
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "results": results,
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
