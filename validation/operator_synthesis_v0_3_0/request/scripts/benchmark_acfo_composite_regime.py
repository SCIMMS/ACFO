from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


try:
    from waxs_cake.composite_wave_operator import PreparedLayeredVectorCompositeOperator
    from waxs_cake.vector_debye import gauss_sine_theta_grid
except ImportError as exc:
    raise RuntimeError(
        "Set PYTHONPATH to the composite component's src directory before running"
    ) from exc


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny)
    )


def build_channels(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    radial = np.sin(theta)[:, None] / np.max(np.sin(theta))
    angle = phi[None, :]
    envelope = np.exp(-0.9 * radial**2)
    channels = np.zeros((6, 2, theta.size, phi.size), dtype=np.complex128)
    channels[0, 0] = envelope
    channels[1, 1] = envelope
    channels[2, 0] = envelope * np.exp(2j * angle)
    channels[3, 1] = envelope * np.exp(-3j * angle)
    channels[4, 0] = 0.64 * envelope * np.exp(4j * angle)
    channels[4, 1] = 0.31j * envelope * np.exp(4j * angle)
    aberration = np.exp(
        1j
        * (
            0.27 * radial**2 * np.cos(2.0 * angle)
            + 0.11 * radial**3 * np.sin(angle)
        )
    )
    channels[5, 0] = 0.48 * envelope * aberration
    channels[5, 1] = 0.56 * envelope * np.exp(1j * angle) * aberration
    return channels


def timed(function, warmup: int, repeats: int) -> tuple[np.ndarray, list[float]]:
    value = function()
    for _ in range(warmup):
        value = function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - start)
    return np.asarray(value), samples


def run_case(case: dict[str, Any], mode_padding: int, miller_margin: int) -> dict[str, Any]:
    ntheta = int(case["ntheta"])
    nphi = int(case["nphi"])
    nrho = int(case["nrho"])
    npsi = int(case["npsi"])
    nz = int(case["nz"])
    channel_count = int(case["channels"])
    warmup = int(case["warmup"])
    repeats = int(case["repeats"])
    if channel_count not in {2, 4, 6}:
        raise ValueError("prespecified channel count must be 2, 4, or 6")

    theta, theta_weights = gauss_sine_theta_grid(ntheta, 1.0)
    phi = 0.07 + np.arange(nphi) * (2.0 * np.pi / nphi)
    source_basis = build_channels(theta, phi)[:channel_count]
    labels = ("x0", "y0", "x+2", "y-3", "elliptic+4", "aberrated")[:channel_count]
    setup_start = time.perf_counter()
    operator = PreparedLayeredVectorCompositeOperator.build(
        theta=theta,
        theta_weights=theta_weights,
        phi=phi,
        rho_axis=np.linspace(0.0, 0.9, nrho),
        psi_axis=np.arange(npsi) * (2.0 * np.pi / npsi),
        z_axis=np.linspace(-0.28, 0.32, nz),
        upper_wavenumber=6.4,
        lower_wavenumber=8.1,
        damping=0.055,
        source_height=0.47,
        lateral_displacement=(0.19, -0.13),
        source_basis=source_basis,
        channel_labels=labels,
        mode_padding=mode_padding,
        miller_margin=miller_margin,
    )
    compact = operator.compact()
    setup_seconds = time.perf_counter() - setup_start
    coefficients = np.asarray(
        [
            0.73 - 0.12j,
            -0.31 + 0.27j,
            0.18 + 0.09j,
            -0.16j,
            0.11 - 0.08j,
            -0.07 + 0.13j,
        ][:channel_count],
        dtype=np.complex128,
    )
    materialized, materialized_samples = timed(
        lambda: compact.forward(coefficients), warmup, repeats
    )
    unfused, unfused_samples = timed(
        lambda: operator.reference_forward(coefficients), warmup, repeats
    )
    error = relative_l2(materialized, unfused)
    materialized_median = float(np.median(materialized_samples))
    unfused_median = float(np.median(unfused_samples))
    saved_per_call = unfused_median - materialized_median
    break_even = setup_seconds / saved_per_call if saved_per_call > 0.0 else None
    return {
        "id": str(case["id"]),
        "grid": {
            "ntheta": ntheta,
            "nphi": nphi,
            "nrho": nrho,
            "npsi": npsi,
            "nz": nz,
            "channels": channel_count,
            "target_count": nrho * npsi * nz,
        },
        "contract": operator.contract.to_dict(),
        "materialized_vs_unfused_relative_l2": error,
        "setup_seconds_conservative": setup_seconds,
        "timing": {
            "warmup": warmup,
            "repeats": repeats,
            "materialized_samples_seconds": materialized_samples,
            "unfused_samples_seconds": unfused_samples,
            "materialized_median_seconds": materialized_median,
            "unfused_median_seconds": unfused_median,
            "unfused_over_materialized_ratio": unfused_median / materialized_median,
            "saved_seconds_per_hot_call": saved_per_call,
            "conservative_break_even_applications": break_even,
            "break_even_note": (
                "The numerator conservatively charges the complete prepared-operator "
                "construction because this implementation does not expose a separate "
                "unfused-chain setup timer."
            ),
        },
        "storage": {
            "materialized_array_bytes": operator.materialized_bytes,
            "retained_unfused_chain_array_bytes": operator.reference_chain_bytes,
            "unfused_over_materialized_ratio": (
                operator.reference_chain_bytes / operator.materialized_bytes
            ),
            "avoided_intermediate_bytes_per_hot_call": (
                operator.avoided_intermediate_bytes_per_call
            ),
        },
        "numerical_passed": error <= 5.0e-13,
        "hot_speed_direction_favourable": saved_per_call > 0.0,
    }


def benchmark(protocol: Path, output: Path, only_case: str | None) -> dict[str, Any]:
    frozen = load_object(protocol)
    cases = frozen.get("prespecified_regime_cases", [])
    if only_case is not None:
        cases = [case for case in cases if str(case.get("id")) == only_case]
        if not cases:
            raise ValueError(f"unknown prespecified case: {only_case}")
    rows = [run_case(case, mode_padding=8, miller_margin=32) for case in cases]
    numerical_passed = bool(rows and all(row["numerical_passed"] for row in rows))
    result = {
        "schema": "acfo-composite-prespecified-regime-map-v1",
        "passed": numerical_passed,
        "scientific_outcome": {
            "all_hot_speed_directions_favourable": bool(
                rows and all(row["hot_speed_direction_favourable"] for row in rows)
            ),
            "performance_sign_is_not_an_integrity_or_numerical_gate": True,
        },
        "protocol": str(protocol),
        "operator_expression": "A_Gamma G_layer T(d)",
        "scope": (
            "Prespecified CPU regime map for the prepared finite action. Ratios are "
            "within-machine comparisons with the algebraically identical unfused chain."
        ),
        "thresholds": {"materialized_vs_unfused_relative_l2_max": 5.0e-13},
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-case", choices=("small", "medium", "large"))
    args = parser.parse_args()
    result = benchmark(args.protocol, args.output, args.only_case)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
