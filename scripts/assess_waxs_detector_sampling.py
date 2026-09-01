from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark_results" / "waxs_detector_sampling_audit.json"


QMIN = 0.05
QMAX = 6.3
NQ = 256
NPHI = 2160


DETECTORS = [
    {
        "name": "EIGER2 X 4M",
        "pixel_pitch_um": 75.0,
        "pixel_array": [2068, 2162],
        "active_area_mm": [155.1, 162.15],
        "representative_distance_mm": 100.0,
        "source": "https://media.dectris.com/TechnicalSpecifications_EIGER2_X_4M_print.pdf",
        "distance_note": "representative WAXS geometry for sampling audit; not a vendor limit",
    },
    {
        "name": "AGIPD 1M",
        "pixel_pitch_um": 200.0,
        "pixel_array": [1024, 1024],
        "active_area_mm": [204.8, 204.8],
        "representative_distance_mm": 125.0,
        "source": "https://www.xfel.eu/facility/instruments/spb_sfx/instrument_parameters_for_2026_01/index_eng.html",
        "distance_note": "official approximate minimum SPB/SFX sample-detector distance",
    },
]


STRUCTURES = [
    "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz",
]


def q_from_angle(alpha: float, wavelength_angstrom: float) -> float:
    return 4.0 * math.pi / wavelength_angstrom * math.sin(0.5 * alpha)


def angle_from_q(q: float, wavelength_angstrom: float) -> float:
    value = q * wavelength_angstrom / (4.0 * math.pi)
    if not 0.0 <= value <= 1.0:
        raise ValueError("q is outside the elastic scattering sphere")
    return 2.0 * math.asin(value)


def detector_case(detector: dict, wavelength_angstrom: float) -> dict:
    pixel_m = detector["pixel_pitch_um"] * 1e-6
    distance_m = detector["representative_distance_mm"] * 1e-3
    active_w_m = detector["active_area_mm"][0] * 1e-3
    active_h_m = detector["active_area_mm"][1] * 1e-3
    half_side_m = 0.5 * min(active_w_m, active_h_m)
    half_corner_m = 0.5 * math.hypot(active_w_m, active_h_m)
    alpha = angle_from_q(QMAX, wavelength_angstrom)
    radius_m = distance_m * math.tan(alpha)
    q_corner = q_from_angle(math.atan2(half_corner_m, distance_m), wavelength_angstrom)
    q_full_circle = q_from_angle(math.atan2(half_side_m, distance_m), wavelength_angstrom)
    dq = (QMAX - QMIN) / (NQ - 1)
    dq_per_pixel_center = 2.0 * math.pi / wavelength_angstrom * pixel_m / distance_m
    dq_dalpha_outer = 2.0 * math.pi / wavelength_angstrom * math.cos(0.5 * alpha)
    dalpha_dr_outer = distance_m / (distance_m * distance_m + radius_m * radius_m)
    dq_per_pixel_outer = dq_dalpha_outer * dalpha_dr_outer * pixel_m
    azimuthal_arc_per_bin_m = radius_m * 2.0 * math.pi / NPHI
    raw_pixels = int(detector["pixel_array"][0] * detector["pixel_array"][1])
    return {
        "detector": detector["name"],
        "wavelength_angstrom": wavelength_angstrom,
        "photon_energy_keV": 12.39842 / wavelength_angstrom,
        "sample_detector_distance_mm": detector["representative_distance_mm"],
        "qmax_two_theta_deg": math.degrees(alpha),
        "qmax_radius_mm": radius_m * 1e3,
        "detector_half_side_mm": half_side_m * 1e3,
        "detector_half_corner_mm": half_corner_m * 1e3,
        "qmax_at_corner_inv_angstrom": q_corner,
        "qmax_full_azimuth_inv_angstrom": q_full_circle,
        "requested_qmax_reaches_corner": QMAX <= q_corner,
        "requested_qmax_has_full_azimuth": QMAX <= q_full_circle,
        "radial_pixels_per_Nq_bin_center": dq / dq_per_pixel_center,
        "radial_pixels_per_Nq_bin_at_qmax": dq / dq_per_pixel_outer,
        "azimuthal_pixels_per_Nphi_bin_at_qmax": azimuthal_arc_per_bin_m / pixel_m,
        "raw_detector_pixels": raw_pixels,
        "polar_target_fraction_of_raw_pixels": (NQ * NPHI) / raw_pixels,
    }


def structure_case(relative_path: str) -> dict:
    path = ROOT / relative_path
    with np.load(path) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        span_nm = np.asarray(metadata["extent_nm"]["span"], dtype=np.float64)
    maximum_extent_nm = float(np.max(span_nm))
    fringe_period_inv_angstrom = 2.0 * math.pi / (10.0 * maximum_extent_nm)
    nyquist_step_inv_angstrom = math.pi / (10.0 * maximum_extent_nm)
    dq = (QMAX - QMIN) / (NQ - 1)
    return {
        "structure_path": relative_path,
        "structure_id": metadata["structure_id"],
        "atoms": metadata["n_atoms"],
        "supercell": metadata["supercell"],
        "extent_nm": span_nm.tolist(),
        "maximum_extent_nm": maximum_extent_nm,
        "finite_size_fringe_period_2pi_over_D_inv_angstrom": fringe_period_inv_angstrom,
        "nyquist_step_pi_over_D_inv_angstrom": nyquist_step_inv_angstrom,
        "current_dq_over_fringe_period": dq / fringe_period_inv_angstrom,
        "current_dq_over_nyquist_step": dq / nyquist_step_inv_angstrom,
        "Nq_needed_for_one_sample_per_fringe": math.ceil(
            (QMAX - QMIN) / fringe_period_inv_angstrom
        )
        + 1,
        "Nq_needed_for_two_samples_per_fringe": math.ceil(
            (QMAX - QMIN) / nyquist_step_inv_angstrom
        )
        + 1,
    }


def main() -> None:
    dq = (QMAX - QMIN) / (NQ - 1)
    detector_rows = [
        detector_case(detector, wavelength)
        for wavelength in (1.0, 0.8)
        for detector in DETECTORS
    ]
    structure_rows = [structure_case(path) for path in STRUCTURES]
    result = {
        "schema": "waxs-detector-sampling-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_grid": {
            "qmin_inv_angstrom": QMIN,
            "qmax_inv_angstrom": QMAX,
            "Nq": NQ,
            "dq_inv_angstrom": dq,
            "Nphi": NPHI,
            "dphi_deg": 360.0 / NPHI,
            "polar_targets": NQ * NPHI,
        },
        "detector_definitions": DETECTORS,
        "detector_geometry_rows": detector_rows,
        "nanocrystal_sampling_rows": structure_rows,
        "decision": {
            "Nq256_detector_realism": (
                "realistic, not artificially dense: for the representative 15.5-keV geometries, "
                "one radial q bin spans about 2-10 detector pixels"
            ),
            "Nphi2160_detector_realism": (
                "realistic at the outer WAXS ring: about 2 AGIPD pixels or 4 EIGER pixels per azimuthal bin"
            ),
            "qmax_geometry_caveat": (
                "at 12.4 keV and the representative distances q=6.3 inverse angstrom lies outside both detector corners; "
                "15.5 keV reaches the corners but not full azimuth, so a detector mask/partial-arc contract is required"
            ),
            "nanocrystal_resolution_caveat": (
                "Nq=256 is marginal or coarse for coherent finite-domain fringes: the 3x3x3 and 5x5x5 structures "
                "have 2pi/D periods of about 0.0200 and 0.0119 inverse angstrom"
            ),
            "recommended_benchmark_contract": (
                "retain Nq=256 as a minimum detector-realistic case, add Nq=512 and preferably Nq near 1024 for "
                "coherent single-domain speckles, and benchmark the actual masked detector nodes or radial-dependent "
                "polar occupancy rather than a full rectangular q-phi grid alone"
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
