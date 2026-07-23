from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    BinnedStructure,
    PreparedAxisymmetricOperator,
    binned_structure_grid,
    direct_axisymmetric_adjoint,
    prepare_axisymmetric_plan,
)
try:
    from scripts.validate_axisymmetric_manifold_discrete import make_curvature_family  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    from validate_axisymmetric_manifold_discrete import make_curvature_family  # type: ignore[no-redef]  # noqa: E402


def make_operator_problem(seed: int = 20260711) -> tuple[BinnedStructure, np.random.Generator]:
    rng = np.random.default_rng(seed)
    n_r, n_z, n_phi = 4, 3, 32
    r_edges = np.linspace(0.0, 2.0, n_r + 1)
    z_edges = np.linspace(-1.2, 1.2, n_z + 1)
    beta_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    histogram = (
        rng.normal(size=(2, n_r, n_z, n_phi))
        + 1j * rng.normal(size=(2, n_r, n_z, n_phi))
    )
    template = BinnedStructure(
        hist=histogram,
        r_centers=0.5 * (r_edges[:-1] + r_edges[1:]),
        z_centers=0.5 * (z_edges[:-1] + z_edges[1:]),
        beta_centers=0.5 * (beta_edges[:-1] + beta_edges[1:]),
        elements=("A", "B"),
        r_edges=r_edges,
        z_edges=z_edges,
        beta_edges=beta_edges,
    )
    return template, rng


def form_factors() -> dict[str, object]:
    return {
        "A": lambda q: 1.0 + 0.08 * q + 0.05j,
        "B": lambda q: 0.72 - 0.04 * q + 0.13j,
    }


def weighted_manifold(manifold: AxisymmetricManifold) -> AxisymmetricManifold:
    phase = np.linspace(0.0, np.pi, manifold.n_u)
    weights = 0.35 + 0.9 * np.sin(phase) ** 2 + 0.15 * np.linspace(0.0, 1.0, manifold.n_u)
    return AxisymmetricManifold(
        manifold.u,
        manifold.q_perp,
        manifold.q_z,
        data_weights=weights,
        name=manifold.name,
        interpretation=manifold.interpretation,
    )


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((actual - reference).ravel()) / np.linalg.norm(reference.ravel()))


def validate() -> dict[str, dict[str, float]]:
    template, rng = make_operator_problem()
    coords, elements = binned_structure_grid(template)
    factors = form_factors()
    results: dict[str, dict[str, float]] = {}
    for family, base_manifold in make_curvature_family(n_u=12).items():
        manifold = weighted_manifold(base_manifold)
        operator = PreparedAxisymmetricOperator(
            template,
            manifold,
            form_factors=factors,
            complex_dtype=np.complex128,
        )
        object_values = np.asarray(template.hist, dtype=np.complex128)
        data_values = (
            rng.normal(size=operator.data_shape)
            + 1j * rng.normal(size=operator.data_shape)
        )

        forward = operator.forward(object_values)
        legacy = prepare_axisymmetric_plan(
            template,
            manifold,
            form_factors=factors,
            circular_backend="numpy",
            complex_dtype=np.complex128,
        ).circular_fft()
        euclidean_adjoint = operator.adjoint_euclidean(data_values)
        weighted_adjoint = operator.adjoint_weighted(data_values)
        direct_euclidean = direct_axisymmetric_adjoint(
            coords,
            manifold,
            operator.phi,
            data_values,
            elements=elements,
            form_factors=factors,
        ).reshape(operator.object_shape)
        direct_weighted = direct_axisymmetric_adjoint(
            coords,
            manifold,
            operator.phi,
            data_values,
            elements=elements,
            form_factors=factors,
            data_weights=manifold.resolved_data_weights,
        ).reshape(operator.object_shape)

        results[family] = {
            "forward_vs_legacy_relative_l2": relative_l2(forward, legacy),
            "euclidean_adjoint_vs_direct_relative_l2": relative_l2(
                euclidean_adjoint,
                direct_euclidean,
            ),
            "weighted_adjoint_vs_direct_relative_l2": relative_l2(
                weighted_adjoint,
                direct_weighted,
            ),
            "euclidean_dot_product_error": operator.adjoint_test(
                object_values,
                data_values,
            ),
            "weighted_dot_product_error": operator.adjoint_test(
                object_values,
                data_values,
                weighted=True,
            ),
            "weighted_vs_euclidean_adjoint_relative_l2": relative_l2(
                weighted_adjoint,
                euclidean_adjoint,
            ),
        }
    return results


def render_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, dict)
    lines = [
        "# ACFO stage-4 forward-adjoint validation",
        "",
        "A complex two-element cylindrical object, complex q-dependent form factors, and complex random data are used. The prepared adjoint is compared with an independent Cartesian conjugate exponent sum.",
        "",
        "| family | forward vs legacy | Euclidean adjoint vs direct | weighted adjoint vs direct | Euclidean dot error | weighted dot error | weighted vs Euclidean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, raw in results.items():
        lines.append(
            f"| {family} | {raw['forward_vs_legacy_relative_l2']:.3e} | "
            f"{raw['euclidean_adjoint_vs_direct_relative_l2']:.3e} | "
            f"{raw['weighted_adjoint_vs_direct_relative_l2']:.3e} | "
            f"{raw['euclidean_dot_product_error']:.3e} | "
            f"{raw['weighted_dot_product_error']:.3e} | "
            f"{raw['weighted_vs_euclidean_adjoint_relative_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "The forward remains an unweighted point-evaluation operator. The weighted adjoint is `A^H W`; no surface Jacobian is applied unless supplied explicitly through the manifold data weights.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_adjoint.py",
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
        default=ROOT / "benchmark_results" / "acfo_stage4_adjoint_validation.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_stage4_adjoint_validation.md",
    )
    args = parser.parse_args()

    results = validate()
    tolerance = 1e-12
    weighted_difference_min = 1e-3
    passed = all(
        raw["forward_vs_legacy_relative_l2"] <= tolerance
        and raw["euclidean_adjoint_vs_direct_relative_l2"] <= tolerance
        and raw["weighted_adjoint_vs_direct_relative_l2"] <= tolerance
        and raw["euclidean_dot_product_error"] <= tolerance
        and raw["weighted_dot_product_error"] <= tolerance
        and raw["weighted_vs_euclidean_adjoint_relative_l2"] >= weighted_difference_min
        for raw in results.values()
    )
    payload: dict[str, object] = {
        "schema": "acfo-stage4-adjoint-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "inner_products": {
            "object": "Euclidean over discrete cylindrical coefficients",
            "data_euclidean": "Euclidean over (u, phi) samples",
            "data_weighted": "radial data_weights[u], broadcast over phi",
        },
        "acceptance": {
            "relative_error_max": tolerance,
            "weighted_vs_euclidean_min": weighted_difference_min,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
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
