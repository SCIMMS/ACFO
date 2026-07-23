from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_debye_wolf import (  # noqa: E402
    PreparedSeparableHarmonicDebyeWolfPlan,
    direct_debye_wolf,
    focal_axes,
    flatten_focal_axes,
    gauss_theta_grid,
    pupil_field,
    relative_l2,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    direct_vectorial_debye_wolf,
    richards_wolf_jones_matrix,
    separable_vectorial_evaluate,
    vectorial_pupil_jones,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scalar and vector High-NA SI correspondence at three apertures.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/high_na_si_correspondence.json"))
    parser.add_argument("--backend", choices=("auto", "numpy", "cpp"), default="auto")
    args = parser.parse_args()

    sin_theta_values = (0.5, 0.8, 0.95)
    ntheta = 32
    nphi = 96
    k = 2.0 * np.pi
    rho_axis, psi_axis, z_axis = focal_axes(
        nrho=12,
        npsi=24,
        nz=5,
        rho_max=2.0,
        z_max=1.0,
    )
    rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    rows: list[dict[str, object]] = []
    cpp_available = False
    try:
        from waxs_cake import _cpp_high_na  # noqa: F401

        cpp_available = True
    except ImportError:
        pass

    for sin_theta_max in sin_theta_values:
        theta_max = float(np.arcsin(sin_theta_max))
        theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
        scalar_pupil = pupil_field(
            "mixed",
            theta,
            phi,
            theta_max=theta_max,
            strength=0.45,
            vortex_charge=6,
            apodization="sqrt-cos",
        )
        vector_pupil = vectorial_pupil_jones(
            "x_vortex",
            theta,
            phi,
            theta_max=theta_max,
            strength=0.45,
            vortex_charge=6,
        )
        mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")

        start = time.perf_counter()
        plan = PreparedSeparableHarmonicDebyeWolfPlan.build(
            nphi,
            theta,
            theta_weights,
            rho_axis,
            psi_axis,
            z_axis,
            k=k,
            h_cutoff=nphi // 2,
            backend=args.backend,
        )
        build_s = time.perf_counter() - start

        start = time.perf_counter()
        scalar_direct = direct_debye_wolf(
            scalar_pupil,
            theta,
            theta_weights,
            phi,
            rho,
            psi,
            z,
            k=k,
        )
        scalar_direct_s = time.perf_counter() - start
        start = time.perf_counter()
        scalar_separable = plan.evaluate(scalar_pupil)
        scalar_separable_s = time.perf_counter() - start

        start = time.perf_counter()
        vector_direct = direct_vectorial_debye_wolf(
            vector_pupil,
            mixing,
            theta,
            theta_weights,
            phi,
            rho,
            psi,
            z,
            k=k,
        )
        vector_direct_s = time.perf_counter() - start
        start = time.perf_counter()
        vector_separable = separable_vectorial_evaluate(plan, vector_pupil, mixing)
        vector_separable_s = time.perf_counter() - start

        scalar_error = relative_l2(scalar_separable, scalar_direct)
        vector_error = relative_l2(vector_separable, vector_direct)
        component_errors = [
            relative_l2(vector_separable[index], vector_direct[index])
            for index in range(3)
        ]
        rows.append(
            {
                "sin_theta_max": sin_theta_max,
                "theta_max_rad": theta_max,
                "theta_max_deg": float(np.degrees(theta_max)),
                "used_modes": plan.used_modes,
                "basis_mib": plan.basis_mib,
                "build_s": build_s,
                "scalar_direct_s": scalar_direct_s,
                "scalar_separable_s": scalar_separable_s,
                "scalar_complex_l2": scalar_error,
                "vector_direct_s": vector_direct_s,
                "vector_separable_s": vector_separable_s,
                "vector_complex_l2": vector_error,
                "vector_component_complex_l2": component_errors,
                "scalar_pass": scalar_error <= 1e-6,
                "vector_pass": vector_error <= 1e-6,
            }
        )

    result = {
        "schema": "high-na-si-correspondence-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "sin_theta_max": list(sin_theta_values),
            "scalar_complex_l2_max": 1e-6,
            "vector_complex_l2_max": 1e-6,
            "reference": "direct Debye-Wolf/Richards-Wolf quadrature on the identical pupil nodes",
        },
        "configuration": {
            "ntheta": ntheta,
            "nphi": nphi,
            "target_shape": [rho_axis.size, psi_axis.size, z_axis.size],
            "target_count": rho.size,
            "rho_max_wavelengths": 2.0,
            "z_max_wavelengths": 1.0,
            "scalar_pupil": "mixed, sqrt-cos apodization",
            "vector_pupil": "x_vortex charge 6, aplanatic Richards-Wolf matrix, sqrt-cos apodization",
            "h_cutoff": nphi // 2,
            "backend_requested": args.backend,
            "cpp_extension_available": cpp_available,
        },
        "rows": rows,
        "passed": all(bool(row["scalar_pass"] and row["vector_pass"]) for row in rows),
        "scope": "SI numerical correspondence; not a novelty, universal PSF replacement, or external-package performance claim",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = [
        "# High-NA SI correspondence",
        "",
        f"- overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        "- gate: scalar and vector complex field L2 <= 1e-6",
        "- reference: direct Debye-Wolf/Richards-Wolf on identical quadrature nodes",
        "",
        "| sin(theta max) | scalar L2 | vector L2 | Ex L2 | Ey L2 | Ez L2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        components = row["vector_component_complex_l2"]
        lines.append(
            f"| {row['sin_theta_max']:.2f} | {row['scalar_complex_l2']:.3e} | {row['vector_complex_l2']:.3e} | "
            f"{components[0]:.3e} | {components[1]:.3e} | {components[2]:.3e} |"
        )
    lines.extend(
        [
            "",
            "This is a correspondence result for SI. The circular-harmonic identity is treated as known prior art and no universal dense Cartesian PSF replacement claim is made.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
