from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import meep as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_uniaxial_smooth_voxel_source_modal as smooth  # noqa: E402
from probe_closed_surface_modal_reciprocity import relative_l2  # noqa: E402
from probe_uniaxial_distributed_source_modal import (  # noqa: E402
    complex_pairs,
    load_point_calibration,
)
from waxs_cake import (  # noqa: E402
    gayer_5mol_mgo_cln_index,
    linbo3_3m_nonlinear_polarization,
)


TEMPERATURE_C = 24.5
PUMP_WAVELENGTH_UM = 1.064
SH_WAVELENGTH_UM = 0.532
SH_FREQUENCY = 1.0
PUMP_FREQUENCY = 0.5
PUMP_SURFACE_PARAMETER_DEG = 38.0
PUMP_AZIMUTH_DEG = 1.875
SOURCE_HALF_WIDTH = 1.2
SOURCE_SIGMAS = np.array((0.45, 0.40, 0.55), dtype=np.float64)
MODE_PHI_DEGREES = (-5.625, 1.875, 9.375)
MODE_PARAMETER_DEGREES = tuple(np.linspace(15.0, 75.0, 9))
REFERENCE_GRID_N = 96
D22_PM_PER_V = 4.08
D31_PM_PER_V = -4.4
D33_PM_PER_V = -25.0


def physical_configuration(
    *,
    pump_surface_parameter_deg: float = PUMP_SURFACE_PARAMETER_DEG,
    pump_azimuth_deg: float = PUMP_AZIMUTH_DEG,
) -> dict[str, object]:
    n_o_pump = gayer_5mol_mgo_cln_index(
        PUMP_WAVELENGTH_UM, "ordinary", temperature_c=TEMPERATURE_C
    )
    n_e_pump = gayer_5mol_mgo_cln_index(
        PUMP_WAVELENGTH_UM, "extraordinary", temperature_c=TEMPERATURE_C
    )
    n_o_sh = gayer_5mol_mgo_cln_index(
        SH_WAVELENGTH_UM, "ordinary", temperature_c=TEMPERATURE_C
    )
    n_e_sh = gayer_5mol_mgo_cln_index(
        SH_WAVELENGTH_UM, "extraordinary", temperature_c=TEMPERATURE_C
    )

    pump_surface_parameter_deg = float(pump_surface_parameter_deg)
    pump_azimuth_deg = float(pump_azimuth_deg)
    if not np.isfinite(pump_surface_parameter_deg) or not 0.0 < pump_surface_parameter_deg < 90.0:
        raise ValueError("pump_surface_parameter_deg must lie strictly between 0 and 90")
    if not np.isfinite(pump_azimuth_deg):
        raise ValueError("pump_azimuth_deg must be finite")
    parameter = np.deg2rad(pump_surface_parameter_deg)
    azimuth = np.deg2rad(pump_azimuth_deg)
    radial = np.array((np.cos(azimuth), np.sin(azimuth), 0.0))
    pump_k0 = 2.0 * np.pi * PUMP_FREQUENCY
    pump_k_perpendicular = pump_k0 * n_e_pump * np.sin(parameter)
    pump_k_z = pump_k0 * n_o_pump * np.cos(parameter)
    pump_wavevector = pump_k_perpendicular * radial + np.array(
        (0.0, 0.0, pump_k_z)
    )
    pump_electric = (
        (pump_k_z / n_o_pump**2) * radial
        - np.array((0.0, 0.0, pump_k_perpendicular / n_e_pump**2))
    )
    pump_electric /= np.linalg.norm(pump_electric)
    nonlinear_polarization = linbo3_3m_nonlinear_polarization(
        pump_electric,
        d22_pm_per_v=D22_PM_PER_V,
        d31_pm_per_v=D31_PM_PER_V,
        d33_pm_per_v=D33_PM_PER_V,
    )
    source_vector = np.real_if_close(nonlinear_polarization).astype(np.float64)
    source_vector /= np.linalg.norm(source_vector)

    return {
        "indices": {
            "n_o_pump": n_o_pump,
            "n_e_pump": n_e_pump,
            "n_o_sh": n_o_sh,
            "n_e_sh": n_e_sh,
        },
        "pump_surface_parameter_deg": pump_surface_parameter_deg,
        "pump_azimuth_deg": pump_azimuth_deg,
        "pump_wavevector": pump_wavevector,
        "pump_electric": pump_electric,
        "nonlinear_polarization_pm_per_v_scale": nonlinear_polarization,
        "source_vector": source_vector,
        "source_carrier": 2.0 * pump_wavevector,
    }


def configure_smooth_source(physical: dict[str, object]) -> None:
    smooth.SOURCE_HALF_WIDTH = SOURCE_HALF_WIDTH
    smooth.SOURCE_SIGMAS = SOURCE_SIGMAS.copy()
    smooth.SOURCE_CARRIER = np.asarray(physical["source_carrier"], dtype=np.float64)
    smooth.MODE_PHI_DEGREES = MODE_PHI_DEGREES
    smooth.REFERENCE_GRID_N = REFERENCE_GRID_N


def pairs(vector: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in np.asarray(vector)]


def fixed_reference_payload(reference: dict[str, object]) -> dict[str, object]:
    return {
        "voxel_count": reference["voxel_count"],
        "nonzero_bins": reference["nonzero_bins"],
        "voxel_integral": [
            float(reference["voxel_integral"].real),
            float(reference["voxel_integral"].imag),
        ],
        "voxel_direct_vs_analytic_relative_l2": reference[
            "voxel_direct_vs_analytic_relative_l2"
        ],
        "acfo_vs_voxel_direct_relative_l2": reference[
            "acfo_vs_voxel_direct_relative_l2"
        ],
        "acfo_vs_analytic_relative_l2": reference[
            "acfo_vs_analytic_relative_l2"
        ],
        "analytic_modal_amplitudes": complex_pairs(
            np.asarray(reference["analytic_modal"])
        ),
        "acfo_modal_amplitudes": complex_pairs(
            np.asarray(reference["acfo_voxel_modal"])
        ),
        "sample_labels": reference["labels"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk 5 mol% MgO:LiNbO3 undepleted-pump SHG polarization gate "
            "against analytic, ACFO, Yee, and closed-surface FDTD references."
        )
    )
    parser.add_argument("--resolutions", default="20,24")
    parser.add_argument("--cell-width", type=float, default=5.0)
    parser.add_argument("--pml-width", type=float, default=0.5)
    parser.add_argument("--monitor-half-widths", default="1.40,1.75")
    parser.add_argument("--until-after-sources", type=float, default=6.0)
    parser.add_argument("--reference-grid-n", type=int, default=REFERENCE_GRID_N)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--point-calibration", type=Path, action="append", default=None
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/linbo3_shg_modal_reciprocity_probe.json"),
    )
    args = parser.parse_args()

    resolutions = tuple(float(value) for value in args.resolutions.split(","))
    monitor_half_widths = tuple(
        float(value) for value in args.monitor_half_widths.split(",")
    )
    if len(monitor_half_widths) != 2 or not monitor_half_widths[0] < monitor_half_widths[1]:
        raise ValueError("monitor-half-widths must contain two increasing values")
    if SOURCE_HALF_WIDTH >= monitor_half_widths[0]:
        raise ValueError("source must lie strictly inside the inner monitor shell")
    if monitor_half_widths[1] >= args.cell_width / 2.0 - args.pml_width:
        raise ValueError("outer monitor must lie strictly inside the non-PML region")
    if args.reference_grid_n != REFERENCE_GRID_N:
        raise ValueError(f"reference-grid-n must remain {REFERENCE_GRID_N}")

    physical = physical_configuration()
    configure_smooth_source(physical)
    indices = dict(physical["indices"])
    source_vector = np.asarray(physical["source_vector"], dtype=np.float64)
    epsilon_perpendicular = float(indices["n_o_sh"]) ** 2
    epsilon_parallel = float(indices["n_e_sh"]) ** 2
    reference = smooth.fixed_continuous_reference(
        source_vector,
        frequency=SH_FREQUENCY,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        grid_n=args.reference_grid_n,
    )

    base = {
        "schema": "linbo3-undepleted-shg-modal-reciprocity-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "homogeneous lossless bulk 5 mol% MgO:LiNbO3; 1064-to-532-nm "
            "extraordinary undepleted pump; prescribed compact C1 pump-squared "
            "polarization; analytic, fixed-voxel, exact PyMeep Yee-source, ACFO, "
            "and closed-surface FDTD modal references"
        ),
        "claim_boundary": (
            "validates relative complex SH modal amplitudes for an impressed chi2 "
            "polarization; excludes self-consistent pump propagation/depletion, "
            "interfaces, periodic poling/domain walls, material loss, and absolute "
            "unit-power conversion efficiency"
        ),
        "normalization_contract": (
            "the 3m tensor fixes the nonlinear-polarization direction and relative "
            "components; one global source scale is removed; no distributed-source "
            "fit; independently frozen point-source complex gain per resolution/shell"
        ),
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": TEMPERATURE_C,
            "pump_wavelength_um": PUMP_WAVELENGTH_UM,
            "sh_wavelength_um": SH_WAVELENGTH_UM,
            **indices,
            "epsilon_perpendicular_sh": epsilon_perpendicular,
            "epsilon_parallel_sh": epsilon_parallel,
            "d22_pm_per_v": D22_PM_PER_V,
            "d31_pm_per_v": D31_PM_PER_V,
            "d33_pm_per_v": D33_PM_PER_V,
            "sellmeier_reference": (
                "Gayer et al., Applied Physics B 91, 343-348 (2008), "
                "DOI 10.1007/s00340-008-2998-2"
            ),
            "nonlinear_reference": (
                "Shoji/Eckardt coefficient convention; d31 magnitude cross-check: "
                "Chen et al., Optics Communications 274, 213-217 (2007), "
                "DOI 10.1016/j.optcom.2007.02.003"
            ),
        },
        "pump_and_source": {
            "pump_branch": "extraordinary",
            "pump_surface_parameter_deg": PUMP_SURFACE_PARAMETER_DEG,
            "pump_azimuth_deg": PUMP_AZIMUTH_DEG,
            "pump_wavevector_normalized": np.asarray(
                physical["pump_wavevector"]
            ).tolist(),
            "pump_electric_unit_vector": np.asarray(
                physical["pump_electric"]
            ).tolist(),
            "nonlinear_polarization_pm_per_v_scale": pairs(
                np.asarray(physical["nonlinear_polarization_pm_per_v_scale"])
            ),
            "normalized_source_vector": source_vector.tolist(),
            "source_carrier_2k_pump": np.asarray(
                physical["source_carrier"]
            ).tolist(),
            "polarization_envelope": (
                "exp(-sum(x_i^2/(2 sigma_i^2))) times product cos^2(pi x_i/(2h)); "
                "equal to the square of a tapered Gaussian pump envelope"
            ),
            "source_half_width_normalized": SOURCE_HALF_WIDTH,
            "source_half_width_um": SOURCE_HALF_WIDTH * SH_WAVELENGTH_UM,
            "polarization_sigmas_normalized": SOURCE_SIGMAS.tolist(),
            "polarization_sigmas_um": (SOURCE_SIGMAS * SH_WAVELENGTH_UM).tolist(),
            "pump_field_sigmas_um": (
                np.sqrt(2.0) * SOURCE_SIGMAS * SH_WAVELENGTH_UM
            ).tolist(),
        },
        "configuration": {
            "resolutions": resolutions,
            "frequency": SH_FREQUENCY,
            "epsilon_perpendicular": epsilon_perpendicular,
            "epsilon_parallel": epsilon_parallel,
            "source_vector": source_vector.tolist(),
            "cell_width": args.cell_width,
            "pml_width": args.pml_width,
            "monitor_half_widths": monitor_half_widths,
            "until_after_sources": args.until_after_sources,
            "mode_phi_degrees": MODE_PHI_DEGREES,
            "mode_parameter_degrees": MODE_PARAMETER_DEGREES,
            "reference_grid_n": args.reference_grid_n,
            "ordinary_mode_count": len(MODE_PHI_DEGREES) * len(MODE_PARAMETER_DEGREES),
            "extraordinary_mode_count": len(MODE_PHI_DEGREES) * len(MODE_PARAMETER_DEGREES),
        },
        "fixed_reference": fixed_reference_payload(reference),
    }

    reference_gates = {
        "voxel_direct_vs_analytic_l2_le_1pct": reference[
            "voxel_direct_vs_analytic_relative_l2"
        ]
        <= 0.01,
        "fixed_acfo_vs_analytic_l2_le_2pct": reference[
            "acfo_vs_analytic_relative_l2"
        ]
        <= 0.02,
    }
    if args.reference_only:
        result = {
            **base,
            "reference_only": True,
            "gates": reference_gates,
            "passed": all(reference_gates.values()),
            "environment": {
                "meep": mp.__version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"wrote {args.output}", flush=True)
        return

    point_calibration_paths = tuple(
        args.point_calibration
        or [Path("benchmark_results/linbo3_point_modal_calibration.json")]
    )
    calibration_gains, calibration_metadata = load_point_calibration(
        point_calibration_paths,
        resolutions,
        frequency=SH_FREQUENCY,
        epsilon_perpendicular=epsilon_perpendicular,
        epsilon_parallel=epsilon_parallel,
        cell_width=args.cell_width,
        pml_width=args.pml_width,
        monitor_half_widths=monitor_half_widths,
        source_vector=source_vector,
    )

    rows: list[dict[str, object]] = []
    calibrated_by_resolution: dict[float, list[np.ndarray]] = {}
    for index, resolution in enumerate(resolutions, start=1):
        print(f"[{index}/{len(resolutions)}] resolution={resolution:g}", flush=True)
        row, calibrated = smooth.run_resolution(
            resolution,
            source_vector,
            reference,
            calibration_gains[resolution],
            frequency=SH_FREQUENCY,
            epsilon_perpendicular=epsilon_perpendicular,
            epsilon_parallel=epsilon_parallel,
            cell_width=args.cell_width,
            pml_width=args.pml_width,
            monitor_half_widths=monitor_half_widths,
            until_after_sources=args.until_after_sources,
        )
        rows.append(row)
        calibrated_by_resolution[resolution] = calibrated

    finest_resolution = max(resolutions)
    finest = rows[resolutions.index(finest_resolution)]
    if len(resolutions) >= 2:
        next_finest_resolution = sorted(resolutions)[-2]
        grid_relative_l2 = relative_l2(
            calibrated_by_resolution[next_finest_resolution][0],
            calibrated_by_resolution[finest_resolution][0],
        )
    else:
        grid_relative_l2 = float("nan")
    gates = {
        **reference_gates,
        "yee_source_vs_analytic_l2_le_3pct": finest["yee_source"][
            "direct_vs_analytic_relative_l2"
        ]
        <= 0.03,
        "yee_acfo_vs_yee_direct_l2_le_3pct": finest["yee_source"][
            "acfo_vs_direct_relative_l2"
        ]
        <= 0.03,
        "blind_fdtd_vs_yee_direct_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_yee_direct_relative_l2"
        ]
        <= 0.05,
        "blind_fdtd_vs_fixed_acfo_l2_le_5pct": finest["shells"][0][
            "fdtd_vs_fixed_acfo_relative_l2"
        ]
        <= 0.05,
        "ordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_analytic_relative_l2"
        ]["ordinary"]
        <= 0.05,
        "extraordinary_l2_le_5pct": finest["shells"][0][
            "branch_fdtd_vs_analytic_relative_l2"
        ]["extraordinary"]
        <= 0.05,
        "monitor_invariance_le_5pct": finest[
            "calibrated_inner_outer_relative_l2"
        ]
        <= 0.05,
        "finest_next_finest_grid_l2_le_5pct": len(resolutions) >= 2
        and grid_relative_l2 <= 0.05,
    }
    result = {
        **base,
        "reference_only": False,
        "point_calibration": calibration_metadata,
        "rows": rows,
        "finest_next_finest_grid_relative_l2": grid_relative_l2,
        "gates": gates,
        "passed": all(gates.values()),
        "environment": {
            "meep": mp.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
