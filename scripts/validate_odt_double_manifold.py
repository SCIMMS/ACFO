from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedDoubleManifoldOperator,
    conjugate_gradient_tikhonov,
    cylindrical_coordinates,
    direct_double_manifold_adjoint,
    direct_double_manifold_forward,
    double_manifold_nodes,
)


class DirectDoubleMatrixOperator:
    def __init__(
        self,
        r,
        z,
        beta,
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
    ) -> None:
        nodes = double_manifold_nodes(
            outgoing,
            incident,
            outgoing_phi,
            incident_phi,
        )
        coords = cylindrical_coordinates(r, z, beta)
        self.matrix = np.exp(1j * (nodes.reshape(-1, 3) @ coords.T))
        self.object_shape = (len(r), len(z), len(beta))
        self.data_shape = nodes.shape[:-1]

    def forward(self, values) -> np.ndarray:
        return (self.matrix @ np.asarray(values).ravel()).reshape(self.data_shape)

    def adjoint_euclidean(self, values) -> np.ndarray:
        return (self.matrix.conj().T @ np.asarray(values).ravel()).reshape(
            self.object_shape
        )


def sphere_branch(u: np.ndarray, k: float, name: str) -> AxisymmetricManifold:
    return AxisymmetricManifold(
        u,
        k * np.sin(u),
        k * np.cos(u),
        name=name,
        interpretation="dispersion-derived",
    )


def ellipsoid_branch(
    u: np.ndarray,
    a: float,
    c: float,
    name: str,
) -> AxisymmetricManifold:
    return AxisymmetricManifold(
        u,
        a * np.sin(u),
        c * np.cos(u),
        name=name,
        interpretation="dispersion-derived",
    )


def make_problem():
    r = np.array([0.4, 1.1])
    z = np.array([-0.8, 0.0, 0.8])
    beta = AxisymmetricManifold.uniform_phi(16)
    outgoing_phi = AxisymmetricManifold.uniform_phi(16)
    incident_phi = AxisymmetricManifold.uniform_phi(8)
    u_out = np.linspace(0.08, 0.95, 4)
    u_in = np.linspace(0.06, 0.75, 3)
    sphere_out = sphere_branch(u_out, 3.1, "sphere-outgoing")
    sphere_in = sphere_branch(u_in, 3.1, "sphere-incident")
    ellipsoid_out = ellipsoid_branch(
        u_out,
        3.4,
        2.7,
        "ellipsoid-outgoing",
    )
    ellipsoid_in = ellipsoid_branch(
        u_in,
        2.8,
        3.2,
        "ellipsoid-incident",
    )
    beta_values = np.asarray(beta)
    object_values = np.zeros((r.size, z.size, beta.size), dtype=np.complex128)
    object_values[0, 0] = (
        1.0
        + 0.30 * np.cos(beta_values - 0.20)
        + 0.10 * np.sin(2.0 * beta_values)
    )
    object_values[1, 1] = 0.70 + 0.20 * np.cos(beta_values + 0.70)
    object_values[0, 2] = 0.40 + 0.15 * np.sin(2.0 * beta_values - 0.30)
    combinations = {
        "sphere_sphere": (sphere_out, sphere_in),
        "sphere_ellipsoid": (sphere_out, ellipsoid_in),
        "ellipsoid_ellipsoid": (ellipsoid_out, ellipsoid_in),
    }
    return r, z, beta, outgoing_phi, incident_phi, object_values, combinations


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def axial_profile(values: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(values) ** 2, axis=(0, 2))


def axial_centroid(values: np.ndarray, z: np.ndarray) -> float:
    profile = axial_profile(values)
    return float(np.dot(profile, z) / np.sum(profile))


def validate_combinations(
    r,
    z,
    beta,
    outgoing_phi,
    incident_phi,
    object_values,
    combinations,
    *,
    harmonic_cutoff: int,
) -> dict[str, object]:
    rng = np.random.default_rng(20260712)
    results: dict[str, object] = {}
    for name, (outgoing, incident) in combinations.items():
        setup_start = perf_counter()
        operator = PreparedDoubleManifoldOperator(
            r,
            z,
            beta,
            outgoing,
            incident,
            outgoing_phi,
            incident_phi,
            harmonic_cutoff=harmonic_cutoff,
        )
        setup_seconds = perf_counter() - setup_start
        direct_start = perf_counter()
        direct = direct_double_manifold_forward(
            object_values,
            r,
            z,
            beta,
            outgoing,
            incident,
            outgoing_phi,
            incident_phi,
        )
        direct_seconds = perf_counter() - direct_start
        structured_start = perf_counter()
        structured = operator.forward(object_values)
        structured_seconds = perf_counter() - structured_start
        data = rng.normal(size=operator.data_shape) + 1j * rng.normal(
            size=operator.data_shape
        )
        direct_adjoint = direct_double_manifold_adjoint(
            data,
            r,
            z,
            beta,
            outgoing,
            incident,
            outgoing_phi,
            incident_phi,
        )
        structured_adjoint = operator.adjoint_euclidean(data)
        results[name] = {
            "outgoing": outgoing.name,
            "incident": incident.name,
            "data_shape": list(operator.data_shape),
            "direct_forward_relative_l2": relative_l2(structured, direct),
            "direct_adjoint_relative_l2": relative_l2(
                structured_adjoint,
                direct_adjoint,
            ),
            "dot_product_error": operator.adjoint_test(object_values, data),
            "harmonic_cutoff": harmonic_cutoff,
            "prepared_bytes": operator.prepared_bytes,
            "setup_seconds": setup_seconds,
            "structured_forward_seconds": structured_seconds,
            "direct_forward_seconds": direct_seconds,
        }
    return results


def cutoff_sweep(
    r,
    z,
    beta,
    outgoing_phi,
    incident_phi,
    object_values,
    outgoing,
    incident,
) -> list[dict[str, float | int]]:
    reference = direct_double_manifold_forward(
        object_values,
        r,
        z,
        beta,
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
    )
    rows = []
    for cutoff in (6, 10, 14):
        operator = PreparedDoubleManifoldOperator(
            r,
            z,
            beta,
            outgoing,
            incident,
            outgoing_phi,
            incident_phi,
            harmonic_cutoff=cutoff,
        )
        rows.append(
            {
                "harmonic_cutoff": cutoff,
                "relative_l2": relative_l2(operator.forward(object_values), reference),
                "prepared_bytes": operator.prepared_bytes,
            }
        )
    return rows


def inverse_validation(
    r,
    z,
    beta,
    outgoing_phi,
    incident_phi,
    object_values,
    outgoing,
    incident,
) -> dict[str, object]:
    rng = np.random.default_rng(20260713)
    setup_start = perf_counter()
    structured = PreparedDoubleManifoldOperator(
        r,
        z,
        beta,
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
        harmonic_cutoff=14,
    )
    structured_setup = perf_counter() - setup_start
    setup_start = perf_counter()
    direct = DirectDoubleMatrixOperator(
        r,
        z,
        beta,
        outgoing,
        incident,
        outgoing_phi,
        incident_phi,
    )
    direct_setup = perf_counter() - setup_start

    clean = direct.forward(object_values)
    relative_noise = 1e-4
    noise = rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    noise *= relative_noise * np.linalg.norm(clean) / np.linalg.norm(noise)
    data = clean + noise
    regularization = 1e-5 * data.size
    solver_kwargs = {
        "regularization": regularization,
        "max_iterations": 120,
        "relative_tolerance": 1e-7,
        "truth": object_values,
    }
    structured_result = conjugate_gradient_tikhonov(
        structured,
        data,
        **solver_kwargs,
    )
    direct_result = conjugate_gradient_tikhonov(
        direct,
        data,
        **solver_kwargs,
    )
    structured_reconstruction = structured_result.reconstruction
    direct_reconstruction = direct_result.reconstruction
    singular_values = np.linalg.svd(direct.matrix, compute_uv=False)
    truth_centroid = axial_centroid(object_values, z)

    def summarize(result, operator, setup_seconds, prepared_bytes):
        reconstruction = result.reconstruction
        prediction = operator.forward(reconstruction)
        return {
            "converged": result.converged,
            "iterations": result.iterations,
            "setup_seconds": setup_seconds,
            "solve_seconds": result.elapsed_seconds,
            "time_to_solution_seconds": setup_seconds + result.elapsed_seconds,
            "reconstruction_relative_l2": relative_l2(
                reconstruction,
                object_values,
            ),
            "relative_data_residual": relative_l2(prediction, data),
            "axial_profile_relative_l2": relative_l2(
                axial_profile(reconstruction),
                axial_profile(object_values),
            ),
            "axial_centroid_bias": axial_centroid(reconstruction, z) - truth_centroid,
            "prepared_bytes": prepared_bytes,
            "iterative_working_set_bytes": result.working_set_bytes,
            "history": list(result.history),
        }

    return {
        "scope": "unrestricted full discrete object grid",
        "object_shape": list(object_values.shape),
        "coefficient_count": int(object_values.size),
        "relative_complex_noise_norm": relative_noise,
        "regularization": regularization,
        "relative_normal_residual_tolerance": 1e-7,
        "matrix_condition_number": float(singular_values[0] / singular_values[-1]),
        "effective_rank_at_relative_1e-8": int(
            np.count_nonzero(singular_values / singular_values[0] > 1e-8)
        ),
        "structured": summarize(
            structured_result,
            structured,
            structured_setup,
            structured.prepared_bytes,
        ),
        "direct_matrix": summarize(
            direct_result,
            direct,
            direct_setup,
            direct.matrix.nbytes,
        ),
        "structured_vs_direct_reconstruction_relative_l2": relative_l2(
            structured_reconstruction,
            direct_reconstruction,
        ),
    }


def render_markdown(payload: dict[str, object]) -> str:
    combinations = payload["combinations"]
    inverse = payload["inverse_reconstruction"]
    assert isinstance(combinations, dict) and isinstance(inverse, dict)
    lines = [
        "# ACFO stage-10 ODT double-manifold validation",
        "",
        "The operator samples `q = Gamma_out(u_out, phi_out) - Gamma_in(u_in, phi_in)`. Both manifolds are absolute wavevector branches. A prepared double Jacobi-Anger factorization is compared with an independent Cartesian type-3 NUDFT and its conjugate exponent-sum adjoint.",
        "",
        "| combination | forward vs direct | adjoint vs direct | dot error | prepared memory |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in combinations.items():
        lines.append(
            f"| {name.replace('_', '-')} | {result['direct_forward_relative_l2']:.3e} | "
            f"{result['direct_adjoint_relative_l2']:.3e} | "
            f"{result['dot_product_error']:.3e} | {result['prepared_bytes']} B |"
        )
    lines.extend(
        [
            "",
            "## Harmonic convergence",
            "",
            "| cutoff | relative L2 | prepared memory |",
            "|---:|---:|---:|",
        ]
    )
    for row in payload["cutoff_sweep"]:
        lines.append(
            f"| {row['harmonic_cutoff']} | {row['relative_l2']:.3e} | {row['prepared_bytes']} B |"
        )
    lines.extend(
        [
            "",
            "## Unrestricted small inverse reconstruction",
            "",
            "| backend | object L2 | data residual | axial profile L2 | axial bias | iterations | solve (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in (
        ("double harmonic", inverse["structured"]),
        ("direct matrix", inverse["direct_matrix"]),
    ):
        lines.append(
            f"| {label} | {result['reconstruction_relative_l2']:.3e} | "
            f"{result['relative_data_residual']:.3e} | "
            f"{result['axial_profile_relative_l2']:.3e} | "
            f"{result['axial_centroid_bias']:.3e} | {result['iterations']} | "
            f"{result['solve_seconds']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- full-grid coefficient count: `{inverse['coefficient_count']}`",
            f"- direct matrix condition number: `{inverse['matrix_condition_number']:.3e}`",
            f"- structured/direct reconstruction mismatch: `{inverse['structured_vs_direct_reconstruction_relative_l2']:.3e}`",
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "This implementation establishes double-manifold correctness and inverse readiness on a small unrestricted discrete grid. The cached mode-pair kernel is a correctness prototype; it is not yet a memory-scalable production ODT backend or a performance claim.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_odt_double_manifold.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage10_odt_double_manifold.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage10_odt_double_manifold.md",
    )
    args = parser.parse_args()

    (
        r,
        z,
        beta,
        outgoing_phi,
        incident_phi,
        object_values,
        combinations,
    ) = make_problem()
    harmonic_cutoff = 14
    combination_results = validate_combinations(
        r,
        z,
        beta,
        outgoing_phi,
        incident_phi,
        object_values,
        combinations,
        harmonic_cutoff=harmonic_cutoff,
    )
    ellipsoid_out, ellipsoid_in = combinations["ellipsoid_ellipsoid"]
    cutoff_results = cutoff_sweep(
        r,
        z,
        beta,
        outgoing_phi,
        incident_phi,
        object_values,
        ellipsoid_out,
        ellipsoid_in,
    )
    inverse = inverse_validation(
        r,
        z,
        beta,
        outgoing_phi,
        incident_phi,
        object_values,
        ellipsoid_out,
        ellipsoid_in,
    )
    acceptance = {
        "forward_relative_l2_max": 1e-9,
        "adjoint_relative_l2_max": 1e-9,
        "dot_product_error_max": 1e-12,
        "cutoff_final_relative_l2_max": 1e-8,
        "inverse_reconstruction_relative_l2_max": 1e-2,
        "inverse_data_residual_max": 5e-4,
        "structured_vs_direct_reconstruction_relative_l2_max": 1e-3,
    }
    passed = bool(
        all(
            result["direct_forward_relative_l2"]
            <= acceptance["forward_relative_l2_max"]
            and result["direct_adjoint_relative_l2"]
            <= acceptance["adjoint_relative_l2_max"]
            and result["dot_product_error"] <= acceptance["dot_product_error_max"]
            for result in combination_results.values()
        )
        and cutoff_results[-1]["relative_l2"]
        <= acceptance["cutoff_final_relative_l2_max"]
        and inverse["structured"]["converged"]
        and inverse["direct_matrix"]["converged"]
        and inverse["structured"]["reconstruction_relative_l2"]
        <= acceptance["inverse_reconstruction_relative_l2_max"]
        and inverse["structured"]["relative_data_residual"]
        <= acceptance["inverse_data_residual_max"]
        and inverse["structured_vs_direct_reconstruction_relative_l2"]
        <= acceptance["structured_vs_direct_reconstruction_relative_l2_max"]
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage10-odt-double-manifold-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "contract": {
            "node_definition": "Gamma_out - Gamma_in",
            "manifold_inputs": "absolute axisymmetric wavevector branches",
            "object_space": "discrete cylindrical coefficients, Euclidean inner product",
            "data_space": "Euclidean over (u_out, u_in, phi_out, phi_in)",
            "reference": "independent Cartesian type-3 NUDFT and conjugate sum",
        },
        "problem": {
            "object_shape": list(object_values.shape),
            "outgoing_u_samples": int(ellipsoid_out.n_u),
            "incident_u_samples": int(ellipsoid_in.n_u),
            "outgoing_phi_samples": int(outgoing_phi.size),
            "incident_phi_samples": int(incident_phi.size),
            "harmonic_cutoff": harmonic_cutoff,
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "combinations": combination_results,
        "cutoff_sweep": cutoff_results,
        "inverse_reconstruction": inverse,
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
