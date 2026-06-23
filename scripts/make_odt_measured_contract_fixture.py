from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake.odt_measured_contract import (  # noqa: E402
    SCHEMA_VERSION,
    OdtMeasuredData,
    save_odt_measured_contract,
    validate_odt_measured_contract,
)


def ring_illum_dirs(n_illum: int, *, polar_deg: float) -> np.ndarray:
    theta = math.radians(float(polar_deg))
    phi = np.linspace(0.0, 2.0 * math.pi, int(n_illum), endpoint=False)
    dirs = np.stack(
        [
            math.sin(theta) * np.cos(phi),
            math.sin(theta) * np.sin(phi),
            np.full_like(phi, math.cos(theta)),
        ],
        axis=1,
    )
    return np.ascontiguousarray(dirs.astype(np.float64))


def pattern_matrix(n_patterns: int, n_illum: int, active: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((int(n_patterns), int(n_illum)), dtype=np.complex64)
    for row in range(int(n_patterns)):
        idx = rng.choice(int(n_illum), size=int(active), replace=False)
        phase = rng.uniform(0.0, 2.0 * math.pi, size=int(active))
        out[row, idx] = np.exp(1j * phase).astype(np.complex64) / math.sqrt(float(active))
    norms = np.linalg.norm(out, axis=1)
    return np.ascontiguousarray(out / np.maximum(norms[:, None], 1e-30))


def make_fixture(args: argparse.Namespace) -> OdtMeasuredData:
    rng = np.random.default_rng(int(args.seed))
    n_illum = int(args.n_illum)
    n_patterns = int(args.n_patterns)
    cap_radial = int(args.cap_radial)
    cap_phi = int(args.cap_phi)
    q_radial = np.linspace(0.0, float(args.detector_na) * float(args.k), cap_radial, dtype=np.float64)
    q_phi = np.linspace(0.0, 2.0 * math.pi, cap_phi, endpoint=False, dtype=np.float64)
    fields: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": args.experiment_type,
        "measurement_model": args.measurement_model,
        "k": float(args.k),
        "units": "um",
        "illum_dirs": ring_illum_dirs(n_illum, polar_deg=args.illumination_angle_deg),
        "detector_origin": np.array([0.0, 0.0, float(args.detector_distance)], dtype=np.float64),
        "detector_u": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "detector_v": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "detector_pixel_size": np.array([0.1, 0.1], dtype=np.float64),
        "detector_distance": float(args.detector_distance),
        "q_layout": "prepared_ring_stack",
        "cap_radial": cap_radial,
        "cap_phi": cap_phi,
        "q_radial": q_radial,
        "q_phi": q_phi,
        "q_z_model": "full_curved_ewald",
    }
    if args.measurement_model == "complex_field":
        real = rng.normal(size=(n_illum, cap_radial, cap_phi)).astype(np.float32)
        imag = rng.normal(size=(n_illum, cap_radial, cap_phi)).astype(np.float32)
        fields["data"] = (real + 1j * imag).astype(np.complex64)
        fields["experiment_type"] = "complex_odt"
    else:
        pattern = pattern_matrix(n_patterns, n_illum, int(args.active_per_pattern), int(args.seed) + 991)
        fields["pattern_matrix"] = pattern
        fields["pattern_model"] = (
            "coherent" if args.measurement_model == "coherent_intensity" else "incoherent"
        )
        data = rng.random(size=(n_patterns, cap_radial, cap_phi), dtype=np.float32)
        fields["data"] = np.ascontiguousarray(data)
    if args.include_mask:
        fields["mask"] = np.ones_like(np.asarray(fields["data"], dtype=np.float32), dtype=np.float32)
    if args.include_variance:
        fields["variance"] = np.full(np.asarray(fields["data"]).shape, 0.01, dtype=np.float32)
    return OdtMeasuredData(fields=fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic ODT measured-data contract fixture.")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "odt_measured_contract_fixture.npz")
    parser.add_argument("--experiment-type", choices=["complex_odt", "fpdt", "fs_odt"], default="fs_odt")
    parser.add_argument(
        "--measurement-model",
        choices=["complex_field", "coherent_intensity", "incoherent_intensity"],
        default="coherent_intensity",
    )
    parser.add_argument("--n-illum", type=int, default=8)
    parser.add_argument("--n-patterns", type=int, default=4)
    parser.add_argument("--active-per-pattern", type=int, default=3)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--k", type=float, default=17.307319527958313)
    parser.add_argument("--detector-na", type=float, default=0.8)
    parser.add_argument("--detector-distance", type=float, default=1000.0)
    parser.add_argument("--illumination-angle-deg", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-mask", action="store_true")
    parser.add_argument("--include-variance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measured = make_fixture(args)
    report = validate_odt_measured_contract(measured)
    report.raise_for_errors()
    save_odt_measured_contract(args.out, measured)
    print(f"wrote {args.out}")
    print(report.to_dict())


if __name__ == "__main__":
    main()
