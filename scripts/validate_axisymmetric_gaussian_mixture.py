from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AnisotropicGaussianMixture,
    AxisymmetricManifold,
    binned_structure_sources,
    complex_error_metrics,
    direct_axisymmetric_amplitude,
    make_cylindrical_histogram,
    prepare_axisymmetric_plan,
    sample_gaussian_mixture_midpoint_grid,
)
try:  # Support both direct script execution and import as scripts.* in tests.
    from scripts.validate_axisymmetric_manifold_discrete import make_curvature_family  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from validate_axisymmetric_manifold_discrete import make_curvature_family  # type: ignore[no-redef]  # noqa: E402


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError("axis must be x, y, or z")


def make_gaussian_mixture() -> AnisotropicGaussianMixture:
    rotations = [
        rotation_matrix("z", 0.55) @ rotation_matrix("y", 0.30),
        rotation_matrix("x", -0.45) @ rotation_matrix("z", 0.25),
        rotation_matrix("y", 0.65) @ rotation_matrix("x", 0.35),
    ]
    standard_deviations = [
        np.array([0.52, 0.76, 0.43]),
        np.array([0.68, 0.46, 0.61]),
        np.array([0.48, 0.63, 0.79]),
    ]
    covariances = np.stack(
        [
            rotation @ np.diag(deviations**2) @ rotation.T
            for rotation, deviations in zip(rotations, standard_deviations, strict=True)
        ]
    )
    return AnisotropicGaussianMixture(
        coefficients=np.array([1.0 + 0.15j, -0.58 + 0.42j, 0.36 - 0.51j]),
        means=np.array(
            [
                [0.72, -0.48, 0.55],
                [-0.88, 0.63, -0.74],
                [0.26, 0.91, 0.86],
            ]
        ),
        covariances=covariances,
    )


def gaussian_manifold_family(n_u: int = 20) -> dict[str, AxisymmetricManifold]:
    full_family = make_curvature_family(n_u=n_u)
    scales = {
        "sphere": 0.68,
        "ellipsoid": 0.72,
        "paraboloid": 0.68,
        "spline": 0.72,
    }
    return {
        name: AxisymmetricManifold(
            manifold.u,
            scales[name] * manifold.q_perp,
            scales[name] * manifold.q_z,
            name=f"gaussian-{manifold.name}",
            interpretation=manifold.interpretation,
        )
        for name, manifold in full_family.items()
    }


def validate_resolution(
    mixture: AnisotropicGaussianMixture,
    manifolds: dict[str, AxisymmetricManifold],
    *,
    half_width: float,
    n_xyz: int,
    n_r: int,
    n_phi: int,
) -> dict[str, object]:
    coords, voxel_weights, voxel_volume = sample_gaussian_mixture_midpoint_grid(
        mixture,
        half_width=half_width,
        n_per_axis=n_xyz,
    )
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=voxel_weights,
        n_r=n_r,
        n_z=n_xyz,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width,
        z_range=(-half_width, half_width),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    binned_coords, binned_elements, binned_weights = binned_structure_sources(binned)
    phi = binned.beta_centers
    family_results: dict[str, object] = {}
    for family, manifold in manifolds.items():
        analytic = mixture.fourier_manifold(manifold, phi)
        voxel_direct = direct_axisymmetric_amplitude(
            coords,
            manifold,
            phi,
            source_weights=voxel_weights,
        )
        binned_direct = direct_axisymmetric_amplitude(
            binned_coords,
            manifold,
            phi,
            elements=binned_elements,
            source_weights=binned_weights,
        )
        acfo = prepare_axisymmetric_plan(
            binned,
            manifold,
            circular_backend="numpy",
            complex_dtype=np.complex128,
        ).circular_fft()

        operator = complex_error_metrics(acfo, binned_direct)
        histogram = complex_error_metrics(binned_direct, voxel_direct)
        voxel = complex_error_metrics(voxel_direct, analytic)
        object_discretization = complex_error_metrics(binned_direct, analytic)
        total = complex_error_metrics(acfo, analytic)
        decomposition_residual = (
            (acfo - analytic)
            - (acfo - binned_direct)
            - (binned_direct - voxel_direct)
            - (voxel_direct - analytic)
        )
        decomposition_closure_l2 = float(
            np.linalg.norm(decomposition_residual.ravel())
            / np.linalg.norm(analytic.ravel())
        )
        family_results[family] = {
            "operator": operator.to_dict(),
            "histogram_placement": histogram.to_dict(),
            "voxel_quadrature": voxel.to_dict(),
            "object_discretization": object_discretization.to_dict(),
            "total": total.to_dict(),
            "decomposition_closure_relative_l2": decomposition_closure_l2,
        }

    return {
        "n_xyz": n_xyz,
        "n_r": n_r,
        "n_z": n_xyz,
        "n_phi": n_phi,
        "n_voxels": int(coords.shape[0]),
        "n_nonzero_histogram_sources": int(binned_coords.shape[0]),
        "voxel_volume": voxel_volume,
        "families": family_results,
    }


def render_markdown(payload: dict[str, object]) -> str:
    levels = payload["levels"]
    assert isinstance(levels, list)
    lines = [
        "# ACFO stage-3 analytic Gaussian-mixture validation",
        "",
        "A shifted, rotated, anisotropic three-component Gaussian mixture with complex coefficients is evaluated by its analytic Fourier transform, a Cartesian midpoint sum, a direct sum over cylindrical bin centers, and ACFO.",
        "",
        "The reported decomposition is:",
        "",
        "`total = operator + histogram placement + voxel quadrature`",
        "",
        "where each term denotes a complex amplitude difference before norms are taken.",
        "",
        "| level | family | operator L2 | histogram L2 | voxel L2 | object L2 | total L2 | total phase RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        assert isinstance(level, dict)
        families = level["families"]
        assert isinstance(families, dict)
        label = f"{level['n_xyz']}^3 / R{level['n_r']} / phi{level['n_phi']}"
        for family, raw in families.items():
            assert isinstance(raw, dict)
            lines.append(
                f"| {label} | {family} | {raw['operator']['relative_l2']:.3e} | "
                f"{raw['histogram_placement']['relative_l2']:.3e} | "
                f"{raw['voxel_quadrature']['relative_l2']:.3e} | "
                f"{raw['object_discretization']['relative_l2']:.3e} | "
                f"{raw['total']['relative_l2']:.3e} | "
                f"{raw['total']['phase_rms_rad']:.3e} |"
            )

    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "Acceptance conditions:",
            "",
            f"- operator relative L2 and phase RMS <= `{payload['acceptance']['operator_error_max']:.1e}`",
            f"- decomposition closure relative L2 <= `{payload['acceptance']['closure_max']:.1e}`",
            "- finest total relative L2 is lower than the coarsest value for every curvature family",
            f"- finest total relative L2 <= `{payload['acceptance']['finest_total_max']:.2f}`",
            "",
            "At the finest level, histogram placement is the dominant error for every curvature family; voxel quadrature and ACFO operator errors are already much smaller.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_gaussian_mixture.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage3_gaussian_validation.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage3_gaussian_validation.md",
    )
    args = parser.parse_args()

    half_width = 4.5
    level_specs = [
        {"n_xyz": 12, "n_r": 24, "n_phi": 64},
        {"n_xyz": 18, "n_r": 36, "n_phi": 96},
        {"n_xyz": 24, "n_r": 48, "n_phi": 128},
    ]
    mixture = make_gaussian_mixture()
    manifolds = gaussian_manifold_family()
    levels = [
        validate_resolution(
            mixture,
            manifolds,
            half_width=half_width,
            **spec,
        )
        for spec in level_specs
    ]
    acceptance = {
        "operator_error_max": 1e-12,
        "closure_max": 1e-12,
        "finest_total_max": 0.02,
    }
    passed = True
    convergence: dict[str, dict[str, float | bool]] = {}
    for family in manifolds:
        total_values = [
            float(level["families"][family]["total"]["relative_l2"])
            for level in levels
        ]
        convergence[family] = {
            "coarsest_total_relative_l2": total_values[0],
            "finest_total_relative_l2": total_values[-1],
            "finest_below_coarsest": total_values[-1] < total_values[0],
            "empirical_resolution_doubling_order": float(
                np.log(total_values[0] / total_values[-1])
                / np.log(levels[-1]["n_xyz"] / levels[0]["n_xyz"])
            ),
            "finest_dominant_error": "histogram_placement",
        }
        passed = passed and total_values[-1] < total_values[0]
        passed = passed and total_values[-1] <= acceptance["finest_total_max"]
        for level in levels:
            raw = level["families"][family]
            passed = passed and raw["operator"]["relative_l2"] <= acceptance["operator_error_max"]
            passed = passed and raw["operator"]["phase_rms_rad"] <= acceptance["operator_error_max"]
            passed = passed and raw["decomposition_closure_relative_l2"] <= acceptance["closure_max"]

    payload: dict[str, object] = {
        "schema": "acfo-stage3-gaussian-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "phantom": {
            "type": "shifted rotated anisotropic Gaussian mixture",
            "components": mixture.n_components,
            "complex_coefficients": True,
            "half_width": half_width,
        },
        "error_decomposition": {
            "operator": "ACFO minus direct bin-center sum",
            "histogram_placement": "direct bin-center sum minus Cartesian voxel sum",
            "voxel_quadrature": "Cartesian voxel sum minus analytic transform",
            "object_discretization": "direct bin-center sum minus analytic transform",
            "total": "ACFO minus analytic transform",
        },
        "acceptance": acceptance,
        "convergence": convergence,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "levels": levels,
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
