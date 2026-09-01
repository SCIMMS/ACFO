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

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    gayer_5mol_mgo_cln_index,
    linbo3_3m_nonlinear_polarization,
    make_cylindrical_histogram,
    project_vector_born_field,
    uniaxial_eigenpolarization,
)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((model - reference).ravel()) / denominator)


def midpoint_domain_source(
    *,
    n: int,
    half_width_um: float,
    pump_wave_number_per_um: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    spacing = 2.0 * half_width_um / n
    axis = -half_width_um + (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    mask = (
        (x / (0.82 * half_width_um)) ** 4
        + (y / (0.70 * half_width_um)) ** 4
        + (z / (0.90 * half_width_um)) ** 4
        <= 1.0
    )
    hologram = (
        np.cos(2.3 * x + 1.1 * y - 0.7 * z)
        + 0.55 * np.cos(-0.8 * x + 2.0 * y + 1.3 * z + 0.4)
    )
    domains = np.where(hologram >= 0.0, 1.0, -1.0)
    phase = np.exp(1j * 2.0 * pump_wave_number_per_um * x)
    voxel_volume = spacing**3
    coords = np.column_stack((x[mask], y[mask], z[mask]))
    weights = domains[mask] * phase[mask] * voxel_volume
    return coords, weights.astype(np.complex128), voxel_volume


def outgoing_manifold(
    u: np.ndarray,
    *,
    wavelength_um: float,
    n_o: float,
    n_e: float,
    branch: str,
) -> AxisymmetricManifold:
    k0 = 2.0 * np.pi / wavelength_um
    if branch == "ordinary":
        q_perp = k0 * n_o * np.sin(u)
        q_z = k0 * n_o * np.cos(u)
    elif branch == "extraordinary":
        q_perp = k0 * n_e * np.sin(u)
        q_z = k0 * n_o * np.cos(u)
    else:
        raise ValueError("unknown branch")
    return AxisymmetricManifold(
        u,
        q_perp,
        q_z,
        name=f"5mol-MgO-CLN-SH-{branch}",
        interpretation="dispersion-derived",
        frequency_units="inverse_micrometre",
    )


def direct_binned_fourier(binned, manifold: AxisymmetricManifold, *, chunk: int = 8) -> np.ndarray:
    hist = np.asarray(binned.hist[0], dtype=np.complex128)
    ir, iz, ib = np.nonzero(hist)
    weights = hist[ir, iz, ib]
    beta = binned.beta_centers[ib]
    positions = np.column_stack(
        (
            binned.r_centers[ir] * np.cos(beta),
            binned.r_centers[ir] * np.sin(beta),
            binned.z_centers[iz],
        )
    )
    nodes = manifold.target_nodes(binned.beta_centers).reshape(-1, 3)
    result = np.empty(nodes.shape[0], dtype=np.complex128)
    for start in range(0, nodes.shape[0], chunk):
        stop = min(nodes.shape[0], start + chunk)
        phase = positions @ nodes[start:stop].T
        result[start:stop] = weights @ np.exp(1j * phase)
    return result.reshape(manifold.n_u, binned.beta_centers.size)


def run_case(
    *,
    name: str,
    pump_polarization: np.ndarray,
    pump_index: float,
    outgoing_branch: str,
    n: int,
    half_width_um: float,
    n_phi: int,
    u: np.ndarray,
    wavelength_pump_um: float,
    wavelength_sh_um: float,
    n_o_sh: float,
    n_e_sh: float,
) -> dict[str, object]:
    pump_k = 2.0 * np.pi * pump_index / wavelength_pump_um
    coords, weights, voxel_volume = midpoint_domain_source(
        n=n,
        half_width_um=half_width_um,
        pump_wave_number_per_um=pump_k,
    )
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=weights,
        n_r=n,
        n_z=n,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width_um,
        z_range=(-half_width_um, half_width_um),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    manifold = outgoing_manifold(
        u,
        wavelength_um=wavelength_sh_um,
        n_o=n_o_sh,
        n_e=n_e_sh,
        branch=outgoing_branch,
    )
    started = time.perf_counter()
    acfo_scalar = PreparedAxisymmetricOperator(
        binned,
        manifold,
        complex_dtype=np.complex128,
    ).forward(binned.hist)
    acfo_seconds = time.perf_counter() - started
    started = time.perf_counter()
    direct_scalar = direct_binned_fourier(binned, manifold)
    direct_seconds = time.perf_counter() - started

    if outgoing_branch == "ordinary":
        residue = 1.0 / (2.0 * manifold.q_z)
    else:
        residue = (n_o_sh * n_o_sh) / (2.0 * manifold.q_z)
    eigen = uniaxial_eigenpolarization(
        manifold.q_perp,
        manifold.q_z,
        binned.beta_centers,
        epsilon_parallel=n_e_sh * n_e_sh,
        epsilon_perpendicular=n_o_sh * n_o_sh,
        branch=outgoing_branch,
    )
    source_vector = linbo3_3m_nonlinear_polarization(pump_polarization)
    acfo_vector = project_vector_born_field(
        acfo_scalar * residue[:, None],
        eigen,
        source_vector,
    )
    direct_vector = project_vector_born_field(
        direct_scalar * residue[:, None],
        eigen,
        source_vector,
    )
    return {
        "name": name,
        "pump_polarization": [[float(v.real), float(v.imag)] for v in pump_polarization],
        "pump_index": pump_index,
        "outgoing_branch": outgoing_branch,
        "nonlinear_polarization_pm_per_v": [
            [float(v.real), float(v.imag)] for v in source_vector
        ],
        "active_cartesian_voxels": int(coords.shape[0]),
        "nonzero_cylindrical_bins": int(np.count_nonzero(binned.hist)),
        "voxel_volume_um3": voxel_volume,
        "q_samples": int(manifold.n_u * n_phi),
        "acfo_seconds": acfo_seconds,
        "direct_seconds": direct_seconds,
        "scalar_complex_l2": relative_l2(acfo_scalar, direct_scalar),
        "vector_complex_l2": relative_l2(acfo_vector, direct_vector),
        "vector_norm": float(np.linalg.norm(direct_vector)),
        "passed": relative_l2(acfo_vector, direct_vector) <= 1e-8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a vector-Born SHG gate against direct exponent sums.")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/uniaxial_vector_born_direct_64cubed.json"))
    args = parser.parse_args()

    wavelength_pump_um = 1.064
    wavelength_sh_um = 0.532
    temperature_c = 24.5
    n_o_pump = gayer_5mol_mgo_cln_index(wavelength_pump_um, "ordinary", temperature_c=temperature_c)
    n_e_pump = gayer_5mol_mgo_cln_index(wavelength_pump_um, "extraordinary", temperature_c=temperature_c)
    n_o_sh = gayer_5mol_mgo_cln_index(wavelength_sh_um, "ordinary", temperature_c=temperature_c)
    n_e_sh = gayer_5mol_mgo_cln_index(wavelength_sh_um, "extraordinary", temperature_c=temperature_c)
    u = np.linspace(0.08, 0.78, 12)
    half_width_um = 2.0
    cases = [
        run_case(
            name="extraordinary-pump-to-extraordinary-SH",
            pump_polarization=np.array([0.0, 0.0, 1.0], dtype=np.complex128),
            pump_index=n_e_pump,
            outgoing_branch="extraordinary",
            n=args.n,
            half_width_um=half_width_um,
            n_phi=args.n_phi,
            u=u,
            wavelength_pump_um=wavelength_pump_um,
            wavelength_sh_um=wavelength_sh_um,
            n_o_sh=n_o_sh,
            n_e_sh=n_e_sh,
        ),
        run_case(
            name="ordinary-pump-to-ordinary-SH-control",
            pump_polarization=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            pump_index=n_o_pump,
            outgoing_branch="ordinary",
            n=args.n,
            half_width_um=half_width_um,
            n_phi=args.n_phi,
            u=u,
            wavelength_pump_um=wavelength_pump_um,
            wavelength_sh_um=wavelength_sh_um,
            n_o_sh=n_o_sh,
            n_e_sh=n_e_sh,
        ),
    ]
    result = {
        "schema": "uniaxial-vector-born-direct-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": temperature_c,
            "pump_wavelength_um": wavelength_pump_um,
            "sh_wavelength_um": wavelength_sh_um,
            "n_o_pump": n_o_pump,
            "n_e_pump": n_e_pump,
            "n_o_sh": n_o_sh,
            "n_e_sh": n_e_sh,
            "d22_pm_per_v": 4.08,
            "d31_pm_per_v": -4.4,
            "d33_pm_per_v": -25.0,
            "sellmeier_reference": "Gayer et al., Applied Physics B 91, 343-348 (2008), DOI 10.1007/s00340-008-2998-2",
            "nonlinear_reference": "Shoji/Eckardt values summarized for 5 mol% MgO:LiNbO3 at 1064 nm; d31 magnitude independently measured as 4.5 pm/V by Chen et al., Optics Communications 274, 213-217 (2007), DOI 10.1016/j.optcom.2007.02.003",
        },
        "object": {
            "cartesian_shape": [args.n, args.n, args.n],
            "half_width_um": half_width_um,
            "domain": "off-axis binary two-carrier hologram inside a superellipsoid",
            "cylindrical_shape": [args.n, args.n, args.n_phi],
        },
        "metric_gate": {"vector_complex_l2_max": 1e-8},
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
        "scope": "vector first-Born source/projection and direct exponent-sum gate; not full-wave Maxwell/FDTD",
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
        "# Uniaxial vector-Born direct gate",
        "",
        f"- overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- material: 5 mol% MgO:LiNbO3, `{wavelength_pump_um*1000:.0f} -> {wavelength_sh_um*1000:.0f} nm`",
        f"- indices at {temperature_c} C: pump no/ne `{n_o_pump:.6f}/{n_e_pump:.6f}`, SH no/ne `{n_o_sh:.6f}/{n_e_sh:.6f}`",
        f"- object: `{args.n}^3` off-axis binary domain, cylindrical `{args.n} x {args.n} x {args.n_phi}`",
        "",
        "| case | vector L2 | scalar L2 | ACFO s | direct s |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['name']} | {case['vector_complex_l2']:.3e} | {case['scalar_complex_l2']:.3e} | {case['acfo_seconds']:.3f} | {case['direct_seconds']:.3f} |"
        )
    lines.extend(["", "This validates the vector first-Born algebra only; Maxwell/FDTD remains a separate gate."])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
