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
    CartesianSpectralBornReference,
    PreparedAxisymmetricOperator,
    UniaxialScalarDispersion,
    complex_error_metrics,
    make_cylindrical_histogram,
    sample_gaussian_mixture_midpoint_grid,
)
try:  # Support direct execution and import as scripts.*.
    from scripts.validate_axisymmetric_gaussian_mixture import make_gaussian_mixture  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from validate_axisymmetric_gaussian_mixture import make_gaussian_mixture  # type: ignore[no-redef]  # noqa: E402


def density_cube(mixture, *, half_width: float, n_per_axis: int) -> np.ndarray:
    spacing = 2.0 * half_width / n_per_axis
    axis = -half_width + (np.arange(n_per_axis) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return mixture.density(coords).reshape((n_per_axis,) * 3)


def gain_aligned_metrics(model: np.ndarray, data: np.ndarray) -> dict[str, object]:
    gain = complex(np.vdot(model, data) / np.vdot(model, model))
    metrics = complex_error_metrics(gain * model, data).to_dict()
    metrics["best_complex_gain_real"] = gain.real
    metrics["best_complex_gain_imag"] = gain.imag
    return metrics


def dispersion_residuals(
    dispersion: UniaxialScalarDispersion,
    u: np.ndarray,
) -> dict[str, float]:
    ordinary = dispersion.manifold(u, "ordinary")
    extraordinary = dispersion.manifold(u, "extraordinary")
    ordinary_kz = ordinary.q_z + dispersion.incident_kz
    extraordinary_kz = extraordinary.q_z + dispersion.incident_kz
    ordinary_residual = (
        ordinary.q_perp**2
        + ordinary_kz**2
        - dispersion.epsilon_perpendicular * dispersion.k0**2
    )
    extraordinary_residual = (
        extraordinary.q_perp**2 / dispersion.epsilon_parallel
        + extraordinary_kz**2 / dispersion.epsilon_perpendicular
        - dispersion.k0**2
    )
    return {
        "ordinary_max_absolute": float(np.max(np.abs(ordinary_residual))),
        "extraordinary_max_absolute": float(np.max(np.abs(extraordinary_residual))),
    }


def render_markdown(payload: dict[str, object]) -> str:
    branches = payload["branches"]
    control = payload["wrong_curvature_control"]
    assert isinstance(branches, dict) and isinstance(control, dict)
    lines = [
        "# ACFO stage-8 scalar anisotropic-Born PDE validation",
        "",
        "A uniaxial scalar Helmholtz dispersion model generates an ordinary spherical branch and an extraordinary ellipsoidal branch. The independent reference samples the same continuous 3-D contrast on a Cartesian grid, applies a zero-padded 3-D spectral Born solve, and trilinearly interpolates the Cartesian spectrum. It uses no cylindrical harmonics, Bessel kernels, or ACFO contraction.",
        "",
        "This is the first level of the planned PDE hierarchy: a weak-scattering scalar Born validation. It is not a vector Maxwell or multiple-scattering full-wave result.",
        "",
        "## Cartesian spectral-reference convergence",
        "",
        "| branch | padding | PDE vs analytic L2 | phase RMS (rad) |",
        "|---|---:|---:|---:|",
    ]
    for branch_name, branch in branches.items():
        for level in branch["spectral_levels"]:
            lines.append(
                f"| {branch_name} | {level['padding_factor']} | "
                f"{level['pde_vs_analytic']['relative_l2']:.3e} | "
                f"{level['pde_vs_analytic']['phase_rms_rad']:.3e} |"
            )
    lines.extend(
        [
            "",
            "## Finest-grid ACFO comparison",
            "",
            "| branch | ACFO vs PDE L2 | ACFO vs analytic L2 | PDE vs analytic L2 | error-budget ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for branch_name, branch in branches.items():
        lines.append(
            f"| {branch_name} | {branch['acfo_vs_pde']['relative_l2']:.3e} | "
            f"{branch['acfo_vs_analytic']['relative_l2']:.3e} | "
            f"{branch['finest_pde_vs_analytic']['relative_l2']:.3e} | "
            f"{branch['combined_error_budget_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Wrong-curvature control",
            "",
            f"- extraordinary analytic data vs gain-aligned spherical model: `{control['analytic_wrong_sphere_gain_aligned']['relative_l2']:.3e}`",
            f"- extraordinary PDE data vs correct ellipsoidal ACFO: `{control['correct_ellipsoid_acfo_vs_pde']['relative_l2']:.3e}`",
            f"- extraordinary PDE data vs gain-aligned spherical ACFO: `{control['wrong_sphere_acfo_gain_aligned_vs_pde']['relative_l2']:.3e}`",
            f"- wrong/correct residual ratio: `{control['wrong_to_correct_residual_ratio']:.2f}`",
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "The outgoing scalar pole-residue weight is applied explicitly and identically to each ACFO/reference comparison. The wrong-curvature control uses the extraordinary weight for both models, so the measured residual isolates support geometry rather than a polarization or transfer-weight mismatch.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_anisotropic_born.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage8_anisotropic_born.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage8_anisotropic_born.md",
    )
    args = parser.parse_args()

    half_width = 4.5
    n_per_axis = 24
    n_r = 48
    n_phi = 128
    padding_factors = (2, 4, 8)
    u = np.linspace(0.03, 0.90, 20)
    mixture = make_gaussian_mixture()
    dispersion = UniaxialScalarDispersion(
        epsilon_parallel=2.56,
        epsilon_perpendicular=1.44,
        k0=1.6,
    )

    coords, voxel_weights, voxel_volume = sample_gaussian_mixture_midpoint_grid(
        mixture,
        half_width=half_width,
        n_per_axis=n_per_axis,
    )
    density = density_cube(mixture, half_width=half_width, n_per_axis=n_per_axis)
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=voxel_weights,
        n_r=n_r,
        n_z=n_per_axis,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width,
        z_range=(-half_width, half_width),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    phi = binned.beta_centers
    branch_cache: dict[str, dict[str, object]] = {}
    for branch_name in ("ordinary", "extraordinary"):
        manifold = dispersion.manifold(u, branch_name)
        weight = dispersion.outgoing_residue_weight(u, branch_name)
        analytic = mixture.fourier_manifold(manifold, phi) * weight[:, None]
        acfo = PreparedAxisymmetricOperator(binned, manifold).forward(binned.hist)
        acfo = acfo * weight[:, None]
        branch_cache[branch_name] = {
            "manifold": manifold,
            "weight": weight,
            "analytic": analytic,
            "acfo": acfo,
            "spectral_levels": [],
        }

    for padding_factor in padding_factors:
        reference = CartesianSpectralBornReference(
            density,
            half_width=half_width,
            padding_factor=padding_factor,
        )
        for branch_name, cache in branch_cache.items():
            pde = reference.born_field(
                cache["manifold"],
                phi,
                cache["weight"],
            )
            cache["spectral_levels"].append(
                {
                    "padding_factor": padding_factor,
                    "padded_grid_shape": [reference.padded_n] * 3,
                    "frequency_spacing": float(
                        reference.frequencies[1] - reference.frequencies[0]
                    ),
                    "pde_vs_analytic": complex_error_metrics(
                        pde,
                        cache["analytic"],
                    ).to_dict(),
                }
            )
            cache["finest_pde"] = pde

    branches: dict[str, object] = {}
    for branch_name, cache in branch_cache.items():
        acfo_vs_pde = complex_error_metrics(
            cache["acfo"],
            cache["finest_pde"],
        ).to_dict()
        acfo_vs_analytic = complex_error_metrics(
            cache["acfo"],
            cache["analytic"],
        ).to_dict()
        finest_pde_vs_analytic = cache["spectral_levels"][-1]["pde_vs_analytic"]
        combined_bound = (
            acfo_vs_analytic["relative_l2"]
            + finest_pde_vs_analytic["relative_l2"]
        )
        branches[branch_name] = {
            "manifold": {
                "q_perp_max": float(np.max(cache["manifold"].q_perp)),
                "q_z_min": float(np.min(cache["manifold"].q_z)),
                "outgoing_weight_min": float(np.min(cache["weight"])),
                "outgoing_weight_max": float(np.max(cache["weight"])),
            },
            "spectral_levels": cache["spectral_levels"],
            "acfo_vs_pde": acfo_vs_pde,
            "acfo_vs_analytic": acfo_vs_analytic,
            "finest_pde_vs_analytic": finest_pde_vs_analytic,
            "combined_error_bound_relative_l2": combined_bound,
            "combined_error_budget_ratio": acfo_vs_pde["relative_l2"] / combined_bound,
        }

    extraordinary = branch_cache["extraordinary"]
    wrong_manifold = dispersion.manifold(u, "ordinary")
    extraordinary_weight = extraordinary["weight"]
    analytic_wrong = mixture.fourier_manifold(wrong_manifold, phi)
    analytic_wrong = analytic_wrong * extraordinary_weight[:, None]
    acfo_wrong = PreparedAxisymmetricOperator(binned, wrong_manifold).forward(binned.hist)
    acfo_wrong = acfo_wrong * extraordinary_weight[:, None]
    analytic_wrong_aligned = gain_aligned_metrics(analytic_wrong, extraordinary["analytic"])
    acfo_wrong_aligned = gain_aligned_metrics(acfo_wrong, extraordinary["finest_pde"])
    correct_acfo_vs_pde = branches["extraordinary"]["acfo_vs_pde"]
    wrong_to_correct = (
        acfo_wrong_aligned["relative_l2"] / correct_acfo_vs_pde["relative_l2"]
    )
    wrong_curvature_control = {
        "data_branch": "extraordinary ellipsoid",
        "forced_model_branch": "ordinary sphere",
        "shared_weight": "extraordinary scalar pole residue",
        "analytic_wrong_sphere_raw": complex_error_metrics(
            analytic_wrong,
            extraordinary["analytic"],
        ).to_dict(),
        "analytic_wrong_sphere_gain_aligned": analytic_wrong_aligned,
        "correct_ellipsoid_acfo_vs_pde": correct_acfo_vs_pde,
        "wrong_sphere_acfo_gain_aligned_vs_pde": acfo_wrong_aligned,
        "wrong_to_correct_residual_ratio": wrong_to_correct,
    }

    dispersion_check = dispersion_residuals(dispersion, u)
    acceptance = {
        "dispersion_residual_max": 1e-12,
        "finest_pde_relative_l2_max": 3e-3,
        "finest_pde_phase_rms_max_rad": 5e-3,
        "padding_refinement_ratio_max": 0.1,
        "acfo_vs_pde_relative_l2_max": 1.5e-2,
        "combined_error_budget_ratio_max": 1.05,
        "wrong_curvature_gain_aligned_relative_l2_min": 0.1,
        "wrong_to_correct_residual_ratio_min": 10.0,
    }
    passed = bool(
        max(dispersion_check.values()) <= acceptance["dispersion_residual_max"]
        and all(
            branch["finest_pde_vs_analytic"]["relative_l2"]
            <= acceptance["finest_pde_relative_l2_max"]
            and branch["finest_pde_vs_analytic"]["phase_rms_rad"]
            <= acceptance["finest_pde_phase_rms_max_rad"]
            and branch["spectral_levels"][-1]["pde_vs_analytic"]["relative_l2"]
            / branch["spectral_levels"][0]["pde_vs_analytic"]["relative_l2"]
            <= acceptance["padding_refinement_ratio_max"]
            and branch["acfo_vs_pde"]["relative_l2"]
            <= acceptance["acfo_vs_pde_relative_l2_max"]
            and branch["combined_error_budget_ratio"]
            <= acceptance["combined_error_budget_ratio_max"]
            for branch in branches.values()
        )
        and acfo_wrong_aligned["relative_l2"]
        >= acceptance["wrong_curvature_gain_aligned_relative_l2_min"]
        and wrong_to_correct >= acceptance["wrong_to_correct_residual_ratio_min"]
    )

    payload: dict[str, object] = {
        "schema": "acfo-stage8-anisotropic-born-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "validation_scope": {
            "equation_level": "scalar uniaxial Helmholtz, first Born",
            "independent_reference": "Cartesian zero-padded 3-D FFT plus trilinear interpolation",
            "excluded": [
                "vector polarization coupling",
                "multiple scattering",
                "full-wave FDTD/FEM boundary solve",
            ],
            "fourier_sign": "positive exp(+i q dot r)",
        },
        "medium": {
            "epsilon_parallel": dispersion.epsilon_parallel,
            "epsilon_perpendicular": dispersion.epsilon_perpendicular,
            "k0": dispersion.k0,
            "incident_kz": dispersion.incident_kz,
            "optic_axis": "z",
        },
        "dispersion_residuals": dispersion_check,
        "phantom_and_grid": {
            "phantom": "shifted rotated anisotropic Gaussian mixture",
            "components": mixture.n_components,
            "complex_coefficients": True,
            "half_width": half_width,
            "cartesian_grid": [n_per_axis] * 3,
            "voxel_volume": voxel_volume,
            "cylindrical_bins": [n_r, n_per_axis, n_phi],
            "n_u": int(u.size),
            "u_range_rad": [float(u[0]), float(u[-1])],
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "branches": branches,
        "wrong_curvature_control": wrong_curvature_control,
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
