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
    BinnedStructure,
    PreparedAxisymmetricOperator,
    PreparedFinufftAxisymmetricReference,
    binned_structure_grid,
    conjugate_gradient_tikhonov,
)


class DirectMatrixOperator:
    def __init__(self, template: BinnedStructure, manifold: AxisymmetricManifold) -> None:
        coords, _ = binned_structure_grid(template)
        nodes = manifold.target_nodes(template.beta_centers).reshape(-1, 3)
        self.matrix = np.exp(1j * (nodes @ coords.T))
        self.object_shape = np.asarray(template.hist).shape
        self.data_shape = (manifold.n_u, template.n_phi)

    def forward(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.complex128)
        return (self.matrix @ array.ravel()).reshape(self.data_shape)

    def adjoint_euclidean(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.complex128)
        return (self.matrix.conj().T @ array.ravel()).reshape(self.object_shape)


class FinufftOperatorAdapter:
    def __init__(self, plan: PreparedFinufftAxisymmetricReference, object_shape) -> None:
        self.plan = plan
        self.object_shape = tuple(object_shape)
        self.data_shape = plan.data_shape

    def forward(self, values) -> np.ndarray:
        return self.plan.execute(np.asarray(values, dtype=np.complex128).ravel())

    def adjoint_euclidean(self, values) -> np.ndarray:
        return self.plan.adjoint(values).reshape(self.object_shape)


class HarmonicSubspaceOperator:
    """Unitary restriction to selected support cells and azimuthal modes."""

    def __init__(
        self,
        base_operator,
        full_object_shape,
        max_h: int,
        support_indices,
    ) -> None:
        self.base_operator = base_operator
        self.full_object_shape = tuple(full_object_shape)
        self.support_indices = np.asarray(support_indices, dtype=np.int64)
        if self.support_indices.ndim != 2 or self.support_indices.shape[1] != 2:
            raise ValueError("support_indices must have shape (n_support, 2)")
        n_phi = self.full_object_shape[-1]
        modes = np.rint(np.fft.fftfreq(n_phi) * n_phi).astype(np.int64)
        self.mode_indices = np.flatnonzero(np.abs(modes) <= max_h)
        self.angular_modes = modes[self.mode_indices]
        self.object_shape = (self.support_indices.shape[0], self.mode_indices.size)
        self.data_shape = base_operator.data_shape

    def expand(self, coefficients) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.complex128)
        if values.shape != self.object_shape:
            raise ValueError(f"coefficients must have shape {self.object_shape}")
        spectrum = np.zeros(self.full_object_shape, dtype=np.complex128)
        for support_index, (r_index, z_index) in enumerate(self.support_indices):
            spectrum[0, r_index, z_index, self.mode_indices] = values[support_index]
        return np.fft.ifft(spectrum, axis=-1, norm="ortho")

    def compress(self, full_values) -> np.ndarray:
        values = np.asarray(full_values, dtype=np.complex128)
        if values.shape != self.full_object_shape:
            raise ValueError(f"full values must have shape {self.full_object_shape}")
        spectrum = np.fft.fft(values, axis=-1, norm="ortho")
        coefficients = np.empty(self.object_shape, dtype=np.complex128)
        for support_index, (r_index, z_index) in enumerate(self.support_indices):
            coefficients[support_index] = spectrum[
                0,
                r_index,
                z_index,
                self.mode_indices,
            ]
        return coefficients

    def forward(self, coefficients) -> np.ndarray:
        return self.base_operator.forward(self.expand(coefficients))

    def adjoint_euclidean(self, values) -> np.ndarray:
        return self.compress(self.base_operator.adjoint_euclidean(values))


def make_inverse_phantom() -> BinnedStructure:
    n_r, n_z, n_phi = 3, 4, 32
    r_edges = np.linspace(0.0, 2.4, n_r + 1)
    z_edges = np.linspace(-1.5, 1.5, n_z + 1)
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
    histogram = np.zeros((1, n_r, n_z, n_phi), dtype=np.complex128)
    histogram[0, 0, 1] = (
        1.0
        + 0.30 * np.cos(beta_centers - 0.35)
        + 0.16 * np.sin(2.0 * beta_centers + 0.20)
    )
    histogram[0, 1, 2] = (
        0.72
        + 0.27 * np.cos(beta_centers + 0.85)
        - 0.12 * np.sin(2.0 * beta_centers - 0.40)
    )
    histogram[0, 2, 3] = 0.38 + 0.18 * np.cos(
        2.0 * beta_centers - 0.65
    )
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


def manifolds(n_u: int = 40) -> tuple[AxisymmetricManifold, AxisymmetricManifold]:
    u = np.linspace(0.04, 1.32, n_u)
    correct = AxisymmetricManifold(
        u,
        4.0 * np.sin(u),
        2.2 * (np.cos(u) - 1.0),
        name="inverse-ellipsoid-truth",
        interpretation="dispersion-derived",
    )
    wrong = AxisymmetricManifold(
        u,
        4.0 * np.sin(u),
        4.0 * (np.cos(u) - 1.0),
        name="inverse-forced-sphere",
        interpretation="dispersion-derived",
    )
    return correct, wrong


def axial_profile(values: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(values) ** 2, axis=(0, 1, 3))


def axial_centroid(values: np.ndarray, z_centers: np.ndarray) -> float:
    profile = axial_profile(values)
    return float(np.dot(profile, z_centers) / np.sum(profile))


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def prepared_bytes(operator) -> tuple[int, str]:
    if isinstance(operator, HarmonicSubspaceOperator):
        base_bytes, scope = prepared_bytes(operator.base_operator)
        return (
            base_bytes + operator.mode_indices.nbytes + operator.support_indices.nbytes,
            scope + "; harmonic and support index maps",
        )
    if isinstance(operator, PreparedAxisymmetricOperator):
        arrays = (
            operator.r_centers,
            operator.z_centers,
            operator.phi,
            operator.form_factors,
            operator.z_phase,
            operator.kernel_fft,
        )
        return sum(array.nbytes for array in arrays), "all exposed prepared arrays"
    if isinstance(operator, DirectMatrixOperator):
        return operator.matrix.nbytes, "dense complex matrix"
    if isinstance(operator, FinufftOperatorAdapter):
        plan = operator.plan
        arrays = (
            plan.coords,
            plan.source_x,
            plan.source_y,
            plan.source_z,
            plan._targets,
            plan._target_x,
            plan._target_y,
            plan._target_z,
        )
        return sum(array.nbytes for array in arrays), "coordinate arrays; opaque FINUFFT plan excluded"
    raise TypeError("unknown operator type")


def run_reconstruction(
    label: str,
    operator,
    data: np.ndarray,
    truth_coefficients: np.ndarray,
    truth_full: np.ndarray,
    z_centers: np.ndarray,
    *,
    setup_seconds: float,
    regularization: float,
) -> tuple[np.ndarray, dict[str, object]]:
    # Warm both plans before measuring iterative execution.
    warm_data = operator.forward(np.zeros(operator.object_shape, dtype=np.complex128))
    operator.adjoint_euclidean(warm_data)
    result = conjugate_gradient_tikhonov(
        operator,
        data,
        regularization=regularization,
        max_iterations=120,
        relative_tolerance=1e-10,
        truth=truth_coefficients,
    )
    reconstruction_coefficients = result.reconstruction
    reconstruction = operator.expand(reconstruction_coefficients)
    final_prediction = operator.forward(reconstruction_coefficients)
    truth_profile = axial_profile(truth_full)
    reconstruction_profile = axial_profile(reconstruction)
    truth_centroid = axial_centroid(truth_full, z_centers)
    reconstruction_centroid = axial_centroid(reconstruction, z_centers)
    operator_bytes, memory_scope = prepared_bytes(operator)
    summary = {
        "label": label,
        "converged": result.converged,
        "iterations": result.iterations,
        "setup_seconds": setup_seconds,
        "solve_seconds": result.elapsed_seconds,
        "time_to_solution_seconds": setup_seconds + result.elapsed_seconds,
        "mean_iteration_seconds": float(
            np.mean([entry["iteration_seconds"] for entry in result.history])
        ),
        "regularization": regularization,
        "reconstruction_relative_l2": relative_l2(reconstruction, truth_full),
        "relative_data_residual": relative_l2(final_prediction, data),
        "axial_profile_relative_l2": relative_l2(
            reconstruction_profile,
            truth_profile,
        ),
        "truth_axial_centroid": truth_centroid,
        "reconstruction_axial_centroid": reconstruction_centroid,
        "axial_centroid_bias": reconstruction_centroid - truth_centroid,
        "imaginary_energy_fraction": float(
            np.linalg.norm(reconstruction.imag) ** 2
            / np.linalg.norm(reconstruction) ** 2
        ),
        "prepared_operator_bytes": operator_bytes,
        "prepared_memory_scope": memory_scope,
        "iterative_working_set_bytes": result.working_set_bytes,
        "history": list(result.history),
    }
    return reconstruction_coefficients, summary


def render_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    controls = payload["controls"]
    identifiability = payload["identifiability_control"]
    assert (
        isinstance(results, dict)
        and isinstance(controls, dict)
        and isinstance(identifiability, dict)
    )
    lines = [
        "# ACFO stage-9 inverse reconstruction validation",
        "",
        "A real cylindrical phantom with three known support cells and angular modes |h| <= 2 is reconstructed from complex ellipsoidal-manifold data. Every backend uses the same 15-parameter complex CGLS solve of `(A^H A + lambda I)x = A^H y` and the same stopping rule.",
        "",
        "| backend / curvature | object L2 | data residual | axial profile L2 | axial bias | iterations | solve (s) | prepared memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results.values():
        lines.append(
            f"| {result['label']} | {result['reconstruction_relative_l2']:.3e} | "
            f"{result['relative_data_residual']:.3e} | "
            f"{result['axial_profile_relative_l2']:.3e} | "
            f"{result['axial_centroid_bias']:.3e} | {result['iterations']} | "
            f"{result['solve_seconds']:.3e} | {result['prepared_operator_bytes']} B |"
        )
    lines.extend(
        [
            "",
            "## Cross-backend and wrong-curvature controls",
            "",
            f"- direct reconstruction vs ACFO reconstruction: `{controls['direct_vs_acfo_reconstruction_relative_l2']:.3e}`",
            f"- FINUFFT reconstruction vs ACFO reconstruction: `{controls['finufft_vs_acfo_reconstruction_relative_l2']:.3e}`",
            f"- wrong/correct object-error ratio: `{controls['wrong_to_correct_object_error_ratio']:.2f}`",
            f"- wrong/correct axial-profile-error ratio: `{controls['wrong_to_correct_axial_profile_error_ratio']:.2f}`",
            "",
            "## Identifiability boundary",
            "",
            f"If all 12 radial-axial cells are left unknown while retaining the same |h| <= 2 band, the 60-parameter matrix has condition number `{identifiability['condition_number']:.3e}` and the same ridge solve gives object error `{identifiability['reconstruction_relative_l2']:.3e}` despite data residual `{identifiability['relative_data_residual']:.3e}`. The accepted 15-parameter result therefore establishes prepared inverse use on a known sparse support; it does not establish unconstrained single-manifold 3-D tomography.",
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "The direct matrix is retained only for this small reference problem. FINUFFT prepared-memory reporting excludes its opaque internal plan allocation, which is stated explicitly in the JSON artifact.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_inverse_reconstruction.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage9_inverse_reconstruction.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage9_inverse_reconstruction.md",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(20260711)
    template = make_inverse_phantom()
    truth = np.asarray(template.hist, dtype=np.complex128)
    support_indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    correct_manifold, wrong_manifold = manifolds()

    start = perf_counter()
    acfo_correct_base = PreparedAxisymmetricOperator(template, correct_manifold)
    acfo_correct = HarmonicSubspaceOperator(
        acfo_correct_base,
        truth.shape,
        max_h=2,
        support_indices=support_indices,
    )
    acfo_correct_setup = perf_counter() - start
    truth_coefficients = acfo_correct.compress(truth)
    clean_data = acfo_correct.forward(truth_coefficients)
    noise = rng.normal(size=clean_data.shape) + 1j * rng.normal(size=clean_data.shape)
    relative_noise = 1e-3
    noise *= relative_noise * np.linalg.norm(clean_data) / np.linalg.norm(noise)
    data = clean_data + noise
    regularization = 1e-6 * data.size

    start = perf_counter()
    acfo_wrong = HarmonicSubspaceOperator(
        PreparedAxisymmetricOperator(template, wrong_manifold),
        truth.shape,
        max_h=2,
        support_indices=support_indices,
    )
    acfo_wrong_setup = perf_counter() - start
    start = perf_counter()
    direct_correct = HarmonicSubspaceOperator(
        DirectMatrixOperator(template, correct_manifold),
        truth.shape,
        max_h=2,
        support_indices=support_indices,
    )
    direct_setup = perf_counter() - start
    coords, _ = binned_structure_grid(template)
    start = perf_counter()
    finufft_plan = PreparedFinufftAxisymmetricReference(
        coords,
        correct_manifold,
        template.beta_centers,
        eps=1e-11,
        nthreads=1,
    )
    finufft = HarmonicSubspaceOperator(
        FinufftOperatorAdapter(finufft_plan, truth.shape),
        truth.shape,
        max_h=2,
        support_indices=support_indices,
    )
    finufft_setup = perf_counter() - start

    acfo_reconstruction, acfo_result = run_reconstruction(
        "ACFO / correct ellipsoid",
        acfo_correct,
        data,
        truth_coefficients,
        truth,
        template.z_centers,
        setup_seconds=acfo_correct_setup,
        regularization=regularization,
    )
    wrong_reconstruction, wrong_result = run_reconstruction(
        "ACFO / forced sphere",
        acfo_wrong,
        data,
        truth_coefficients,
        truth,
        template.z_centers,
        setup_seconds=acfo_wrong_setup,
        regularization=regularization,
    )
    direct_reconstruction, direct_result = run_reconstruction(
        "direct dense matrix / correct ellipsoid",
        direct_correct,
        data,
        truth_coefficients,
        truth,
        template.z_centers,
        setup_seconds=direct_setup,
        regularization=regularization,
    )
    finufft_reconstruction, finufft_result = run_reconstruction(
        "FINUFFT type-3 / correct ellipsoid",
        finufft,
        data,
        truth_coefficients,
        truth,
        template.z_centers,
        setup_seconds=finufft_setup,
        regularization=regularization,
    )
    results = {
        "acfo_correct": acfo_result,
        "acfo_wrong_sphere": wrong_result,
        "direct_correct": direct_result,
        "finufft_correct": finufft_result,
    }
    controls = {
        "direct_vs_acfo_reconstruction_relative_l2": relative_l2(
            direct_reconstruction,
            acfo_reconstruction,
        ),
        "finufft_vs_acfo_reconstruction_relative_l2": relative_l2(
            finufft_reconstruction,
            acfo_reconstruction,
        ),
        "wrong_to_correct_object_error_ratio": (
            wrong_result["reconstruction_relative_l2"]
            / acfo_result["reconstruction_relative_l2"]
        ),
        "wrong_to_correct_axial_profile_error_ratio": (
            wrong_result["axial_profile_relative_l2"]
            / acfo_result["axial_profile_relative_l2"]
        ),
        "wrong_to_correct_absolute_axial_bias_ratio": (
            abs(wrong_result["axial_centroid_bias"])
            / max(abs(acfo_result["axial_centroid_bias"]), np.finfo(np.float64).tiny)
        ),
    }
    all_support = np.array(
        [
            [r_index, z_index]
            for r_index in range(truth.shape[1])
            for z_index in range(truth.shape[2])
        ],
        dtype=np.int64,
    )
    unrestricted_acfo = HarmonicSubspaceOperator(
        acfo_correct_base,
        truth.shape,
        max_h=2,
        support_indices=all_support,
    )
    unrestricted_truth = unrestricted_acfo.compress(truth)
    unrestricted_result = conjugate_gradient_tikhonov(
        unrestricted_acfo,
        data,
        regularization=regularization,
        max_iterations=120,
        relative_tolerance=1e-10,
        truth=unrestricted_truth,
    )
    unrestricted_full = unrestricted_acfo.expand(unrestricted_result.reconstruction)
    unrestricted_prediction = unrestricted_acfo.forward(
        unrestricted_result.reconstruction
    )
    unrestricted_direct = HarmonicSubspaceOperator(
        direct_correct.base_operator,
        truth.shape,
        max_h=2,
        support_indices=all_support,
    )
    n_unrestricted = int(np.prod(unrestricted_direct.object_shape))
    identity = np.eye(n_unrestricted, dtype=np.complex128)
    reduced_matrix = np.column_stack(
        [
            unrestricted_direct.forward(
                identity[index].reshape(unrestricted_direct.object_shape)
            ).ravel()
            for index in range(n_unrestricted)
        ]
    )
    singular_values = np.linalg.svd(reduced_matrix, compute_uv=False)
    relative_singular_values = singular_values / singular_values[0]
    identifiability_control = {
        "description": "all radial-axial cells unknown, still restricted to |h| <= 2",
        "coefficient_count": n_unrestricted,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "effective_rank_at_relative_1e-6": int(
            np.count_nonzero(relative_singular_values > 1e-6)
        ),
        "reconstruction_relative_l2": relative_l2(unrestricted_full, truth),
        "relative_data_residual": relative_l2(unrestricted_prediction, data),
        "converged": unrestricted_result.converged,
        "iterations": unrestricted_result.iterations,
        "conclusion": "single-manifold unconstrained radial-axial inversion is ill-conditioned",
    }
    acceptance = {
        "all_converged_required": True,
        "correct_reconstruction_relative_l2_max": 0.2,
        "correct_relative_data_residual_max": 0.02,
        "direct_vs_acfo_reconstruction_relative_l2_max": 1e-8,
        "finufft_vs_acfo_reconstruction_relative_l2_max": 1e-6,
        "wrong_to_correct_object_error_ratio_min": 2.0,
        "wrong_to_correct_axial_profile_error_ratio_min": 2.0,
    }
    passed = bool(
        all(result["converged"] for result in results.values())
        and acfo_result["reconstruction_relative_l2"]
        <= acceptance["correct_reconstruction_relative_l2_max"]
        and acfo_result["relative_data_residual"]
        <= acceptance["correct_relative_data_residual_max"]
        and controls["direct_vs_acfo_reconstruction_relative_l2"]
        <= acceptance["direct_vs_acfo_reconstruction_relative_l2_max"]
        and controls["finufft_vs_acfo_reconstruction_relative_l2"]
        <= acceptance["finufft_vs_acfo_reconstruction_relative_l2_max"]
        and controls["wrong_to_correct_object_error_ratio"]
        >= acceptance["wrong_to_correct_object_error_ratio_min"]
        and controls["wrong_to_correct_axial_profile_error_ratio"]
        >= acceptance["wrong_to_correct_axial_profile_error_ratio_min"]
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage9-inverse-reconstruction-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "problem": {
            "full_object_shape": list(truth.shape),
            "inverse_coefficient_shape": list(truth_coefficients.shape),
            "data_shape": list(clean_data.shape),
            "active_truth_angular_modes": "|h| <= 2",
            "known_support_rz_indices": support_indices.tolist(),
            "truth_curvature": "Q_perp=4 sin(u), Q_z=2.2(cos(u)-1)",
            "wrong_curvature": "Q_perp=4 sin(u), Q_z=4(cos(u)-1)",
            "relative_complex_noise_norm": relative_noise,
            "regularization": regularization,
            "solver": "complex conjugate gradient on Tikhonov normal equations",
            "relative_normal_residual_tolerance": 1e-10,
        },
        "acceptance": acceptance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "finufft_threads": 1,
        },
        "results": results,
        "controls": controls,
        "identifiability_control": identifiability_control,
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
