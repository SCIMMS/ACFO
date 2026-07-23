from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

def mib(elements: int, bytes_per_element: int) -> float:
    return elements * bytes_per_element / 1024**2


def operator_estimate(plan, *, bytes_complex: int, illumination_block: int | None = None) -> dict:
    n_i = plan.n_illum if illumination_block is None else min(plan.n_illum, illumination_block)
    n_r, n_z, n_beta, n_h, n_l = plan.n_r, plan.n_z, plan.n_beta, plan.n_h, plan.n_l
    cap_r, cap_phi = plan.cap_radial, plan.cap_phi
    arrays = {
        "coeff": mib(n_r * n_z * n_beta, bytes_complex),
        "coeff_h_full": mib(n_r * n_z * n_beta, bytes_complex),
        "coeff_sources_rzhl": mib(n_r * n_z * n_h * n_l, bytes_complex),
        "decomposed_irzh": mib(n_i * n_r * n_z * n_h, bytes_complex),
        "data": mib(plan.n_illum * cap_r * cap_phi, bytes_complex),
        "compact_adjoint_irzh": mib(n_i * n_r * n_z * n_h, bytes_complex),
        "adjoint_mixed_rzhl": mib(n_r * n_z * n_h * n_l, bytes_complex),
        "out_h": mib(n_r * n_z * n_beta, bytes_complex),
    }
    forward_peak_lower = (
        arrays["coeff"]
        + arrays["coeff_h_full"]
        + arrays["coeff_sources_rzhl"]
        + arrays["decomposed_irzh"]
        + arrays["data"]
    )
    adjoint_peak_lower = (
        arrays["data"]
        + arrays["compact_adjoint_irzh"]
        + arrays["adjoint_mixed_rzhl"]
        + arrays["out_h"]
    )
    return {
        "n_illum": plan.n_illum,
        "n_r": n_r,
        "n_z": n_z,
        "n_beta": n_beta,
        "n_h": n_h,
        "n_l": n_l,
        "cap_radial": cap_r,
        "cap_phi": cap_phi,
        "illumination_block_for_estimate": n_i,
        "arrays_mib": arrays,
        "forward_peak_lower_bound_mib": forward_peak_lower,
        "adjoint_peak_lower_bound_mib": adjoint_peak_lower,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate current and blocked ODT 256-cubed GPU memory.")
    parser.add_argument("--ring-illum", type=int, default=120)
    parser.add_argument("--illumination-blocks", default="1,2,4,8,16")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/odt_256cubed_memory_estimate.json"),
    )
    args = parser.parse_args()

    k = 17.307319527958313
    detector_na = 0.9240924092409241
    illumination_na = np.sin(np.deg2rad(49.0))
    n_r = n_z = n_beta = cap_radial = cap_phi = 256
    h_margin = 20
    l_margin = 18
    h_cutoff = min(n_beta // 2, int(np.ceil(k * detector_na * 1.0 + h_margin)))
    l_cutoff = min(n_beta // 2 - 1, int(np.ceil(k * illumination_na * 1.0 + l_margin)))
    axis_l_cutoff = min(n_beta // 2 - 1, l_margin)
    n_h = 2 * h_cutoff + 1
    n_l = 2 * l_cutoff + 1
    axis_n_l = 2 * axis_l_cutoff + 1
    bytes_complex = 8 if args.dtype == "complex64" else 16

    def low_memory_basis_mib(n_illum: int, local_n_l: int) -> float:
        complex_elements = (
            n_h * cap_radial * n_r  # radial
            + cap_radial * n_z  # axial
            + 2 * n_h  # mode phase and conjugate
            + 3 * local_n_l * n_r  # transverse layouts/conjugate
            + 3 * n_illum * local_n_l  # psi layouts/conjugate
            + 3 * n_z  # axial phase layouts/conjugate
        )
        integer_bytes = (n_h * local_n_l + n_h) * 8
        return mib(complex_elements, bytes_complex) + integer_bytes / 1024**2

    ring_plan = SimpleNamespace(
        n_illum=args.ring_illum,
        n_r=n_r,
        n_z=n_z,
        n_beta=n_beta,
        n_h=n_h,
        n_l=n_l,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    axis_plan = SimpleNamespace(
        n_illum=1,
        n_r=n_r,
        n_z=n_z,
        n_beta=n_beta,
        n_h=n_h,
        n_l=axis_n_l,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    basis_mib = low_memory_basis_mib(args.ring_illum, n_l) + low_memory_basis_mib(1, axis_n_l)
    blocks = [int(value) for value in args.illumination_blocks.split(",") if value.strip()]
    full = {
        "ring": operator_estimate(ring_plan, bytes_complex=bytes_complex),
        "axis": operator_estimate(axis_plan, bytes_complex=bytes_complex),
    }
    blocked = []
    for block in blocks:
        ring = operator_estimate(
            ring_plan,
            bytes_complex=bytes_complex,
            illumination_block=block,
        )
        axis = operator_estimate(
            axis_plan,
            bytes_complex=bytes_complex,
            illumination_block=block,
        )
        blocked.append({"illumination_block": block, "ring": ring, "axis": axis})

    current_forward_lower = basis_mib + full["ring"]["forward_peak_lower_bound_mib"]
    current_adjoint_lower = basis_mib + full["ring"]["adjoint_peak_lower_bound_mib"]
    result = {
        "schema": "odt-256cubed-memory-estimate-v1",
        "problem": {
            "object_shape": [256, 256, 256],
            "object_bins": 256**3,
            "ring_illum": args.ring_illum,
            "axis_included": True,
            "total_illumination_count": args.ring_illum + 1,
            "detector_shape": [256, 256],
            "total_q_samples": (args.ring_illum + 1) * cap_radial * cap_phi,
            "dtype": args.dtype,
        },
        "dimension_derivation": {
            "k": k,
            "detector_na": detector_na,
            "illumination_na": float(illumination_na),
            "h_margin": h_margin,
            "l_margin": l_margin,
            "h_cutoff": h_cutoff,
            "l_cutoff": l_cutoff,
            "n_h": n_h,
            "n_l": n_l,
        },
        "gpu_basis_mib": basis_mib,
        "full_current_estimate": full,
        "current_ring_forward_peak_lower_bound_including_basis_mib": current_forward_lower,
        "current_ring_adjoint_peak_lower_bound_including_basis_mib": current_adjoint_lower,
        "illumination_block_estimates": blocked,
        "gate_24gib": 24 * 1024,
        "current_forward_within_24gib_lower_bound": current_forward_lower <= 24 * 1024,
        "current_adjoint_within_24gib_lower_bound": current_adjoint_lower <= 24 * 1024,
        "interpretation": [
            "These are live-array lower bounds from the actual H/L decomposition dimensions, not measured process peaks.",
            "Illumination blocking reduces irzh arrays but does not reduce rzhl source/mixed arrays; L or RZ blocking is required if those dominate.",
            "A measured run is allowed only after the lower bound leaves sufficient headroom below the hardware limit.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ODT 256-cubed memory estimate",
        "",
        f"- problem: `256^3`, `{result['problem']['total_illumination_count']}` illuminations, detector `256 x 256`",
        f"- q samples: `{result['problem']['total_q_samples']:,}`",
        f"- decomposition: ring H=`{n_h}`, L=`{n_l}`",
        f"- basis: `{basis_mib:.2f} MiB`",
        f"- current forward live-array lower bound: `{current_forward_lower:.2f} MiB`",
        f"- current adjoint live-array lower bound: `{current_adjoint_lower:.2f} MiB`",
        "",
        "| illumination block | ring forward lower MiB | ring adjoint lower MiB |",
        "|---:|---:|---:|",
    ]
    for item in blocked:
        lines.append(
            f"| {item['illumination_block']} | {basis_mib + item['ring']['forward_peak_lower_bound_mib']:.2f} | {basis_mib + item['ring']['adjoint_peak_lower_bound_mib']:.2f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
