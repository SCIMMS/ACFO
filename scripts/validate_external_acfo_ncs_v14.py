from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENVIRONMENT = ROOT / "benchmark_results/local_prepared_waxs_machine_environment.json"


FILES = {
    "manifest": "manifest_verification.json",
    "environment": "environment.json",
    "build_ext": "build_ext_receipt.json",
    "pytest": "pytest_receipt.json",
    "odt_probe": "odt_banded_cartesian_final_packed_probe.json",
    "waxs_prepared": "waxs_prepared_1m_abba.json",
    "waxs_detector": "waxs_detector_nq512_abba.json",
    "odt_scale": "odt_banded_cartesian_final_packed_full_timing.json",
    "odt_direct_c64": "odt_cufinufft_matched_error_direct_subset_c64.json",
    "odt_direct_c128": "odt_cufinufft_matched_error_direct_subset_c128.json",
    "odt_same_dtype": "odt_same_dtype_abba30.json",
    "odt_matched_full": "odt_cufinufft_matched_c128_full_pair5.json",
    "odt_temporal": "odt_banded_cartesian_temporal_warm_start.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(0.5 * (a + b), np.finfo(float).tiny)


def waxs_prepared_gates(payload: dict[str, Any]) -> dict[str, bool]:
    rows = payload.get("measured_pairs", [])
    ab = [float(row["paired_speedup"]) for row in rows if row.get("order") == "AB"]
    ba = [float(row["paired_speedup"]) for row in rows if row.get("order") == "BA"]
    order_gap = (
        relative_gap(float(np.median(ab)), float(np.median(ba)))
        if ab and ba
        else float("inf")
    )
    speed = payload.get("measured_summary", {}).get("paired_speedup", {})
    return {
        "schema": payload.get("schema")
        == "protein-lattice-prepared-finufft-512-abba-v1",
        "complete_and_internal_pass": payload.get("status") == "complete"
        and bool(payload.get("passed")),
        "prepared_fused_contract": payload.get("contract", {}).get(
            "factorized_backend"
        )
        == "prepared_fused"
        and payload.get("contract", {}).get("lattice_backend") == "separable",
        "warmup_10_measured_30": payload.get("warmup_pairs_completed") == 10
        and payload.get("measured_pairs_completed") == 30,
        "balanced_15_ab_15_ba": len(ab) == 15 and len(ba) == 15,
        "order_gap_le_10pct": order_gap <= 0.10,
        "complex_l2_le_2e_6": payload.get("cross_error", {}).get(
            "complex_l2", float("inf")
        )
        <= 2e-6,
        "legacy_l2_le_1e_12": payload.get("legacy_comparison", {}).get(
            "legacy_vs_prepared_complex_l2", float("inf")
        )
        <= 1e-12,
        "paired_median_ge_3": speed.get("median", 0.0) >= 3.0,
        "paired_p05_ge_3": speed.get("p05", 0.0) >= 3.0,
    }


def waxs_detector_gates(payload: dict[str, Any]) -> dict[str, bool]:
    timing = payload.get("timing_protocol", {})
    acfo_peak = payload.get("acfo_first_peak_rss_mib")
    finufft_peak = payload.get("finufft_first_peak_rss_mib")
    memory_ratio = (
        float(finufft_peak) / float(acfo_peak)
        if finite_positive(acfo_peak) and finite_positive(finufft_peak)
        else 0.0
    )
    return {
        "same_binned_detector_contract": payload.get("source_mode") == "same_binned"
        and payload.get("nq") == 512
        and payload.get("n_phi") == 2250
        and payload.get("detector_label") == "EIGER2_X_4M_15p5keV_100mm",
        "warmup_10_repeat_30_abba": timing.get("warmups_per_method") == 10
        and timing.get("measured_repeats_per_method") == 30
        and timing.get("method_order") == "alternating_ab_ba",
        "complex_l2_le_1e_6": payload.get(
            "complex_l2_acfo_vs_finufft", float("inf")
        )
        <= 1e-6,
        "intensity_row_median_le_1e_3": payload.get(
            "intensity_row_relative_l2_median", float("inf")
        )
        <= 1e-3,
        "intensity_row_p99_le_5e_3": payload.get(
            "intensity_row_relative_l2_p99", float("inf")
        )
        <= 5e-3,
        "speedup_gt_1": payload.get("warm_speedup_finufft_over_acfo", 0.0) > 1.0,
        "whole_process_memory_ratio_ge_4": memory_ratio >= 4.0,
        "all_finite": bool(payload.get("all_finite")),
    }


def evaluate_run(
    run_dir: Path,
    *,
    mode: str,
    allow_reference_machine: bool = False,
    local_environment: Path = LOCAL_ENVIRONMENT,
) -> dict[str, Any]:
    required_names = [
        "manifest",
        "environment",
        "build_ext",
        "pytest",
        "odt_probe",
    ]
    if mode == "full":
        required_names.extend(
            [
                "waxs_prepared",
                "waxs_detector",
                "odt_scale",
                "odt_direct_c64",
                "odt_direct_c128",
                "odt_same_dtype",
                "odt_matched_full",
                "odt_temporal",
            ]
        )
    paths = {name: run_dir / FILES[name] for name in required_names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "schema": "external-acfo-ncs-v14-validation-v1",
            "mode": mode,
            "passed": False,
            "execution_pass": False,
            "missing": missing,
        }

    data = {name: load(path) for name, path in paths.items()}
    environment = data["environment"]
    manifest = data["manifest"]
    pytest_receipt = data["pytest"]
    build_receipt = data["build_ext"]
    probe = data["odt_probe"]
    packages = environment.get("runtime", {}).get("packages", {})
    torch_info = environment.get("torch", {})
    quick_gates = {
        "release_manifest_verified": bool(manifest.get("passed")),
        "environment_schema": environment.get("schema")
        == "acfo-ncs-v14-machine-environment-v1",
        "cuda_available": bool(torch_info.get("cuda_available")),
        "gpu_baseline_packages_present": bool(packages.get("cupy-cuda12x"))
        and bool(packages.get("cufinufft")),
        "cpp_extensions_rebuilt": bool(build_receipt.get("passed")),
        "pytest_passed": bool(pytest_receipt.get("passed")),
        "odt_integrated_probe_passed": probe.get("schema")
        == "odt-banded-cartesian-final-packed-integration-v1"
        and bool(probe.get("passed"))
        and all(probe.get("gates", {}).values()),
    }
    result: dict[str, Any] = {
        "schema": "external-acfo-ncs-v14-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "run_dir": run_dir.as_posix(),
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "quick_gates": quick_gates,
        "package_smoke_pass": all(quick_gates.values()),
    }
    if mode == "quick":
        result["independent_machine_replication_pass"] = None
        result["publication_replication_pass"] = False
        result["execution_pass"] = result["package_smoke_pass"]
        result["passed"] = result["execution_pass"]
        result["claim_boundary"] = [
            "Quick mode proves package integrity, installation, CUDA availability, tests and the integrated ODT accuracy probe only.",
            "Quick mode is not an independent publication-scale timing replication.",
        ]
        return result

    local_env = load(local_environment)
    fingerprints_differ = environment.get("machine_fingerprint_sha256") != local_env.get(
        "machine_fingerprint_sha256"
    )
    prepared_gates = waxs_prepared_gates(data["waxs_prepared"])
    detector_gates = waxs_detector_gates(data["waxs_detector"])

    scale = data["odt_scale"]
    scale_cases = {int(row.get("selected_n_z", -1)): row for row in scale.get("cases", [])}
    scale_gates = {
        "schema": scale.get("schema")
        == "odt-banded-cartesian-final-packed-full-timing-v1",
        "external_accuracy_anchor_hash": scale.get("accuracy_anchor", {}).get("sha256")
        == sha256(paths["odt_probe"]),
        "z64_128_256_present": set(scale_cases) == {64, 128, 256},
        "all_integrated_timings_finite": all(
            finite_positive(row.get("integrated_steady_remap_mode_normal_pair_timing", {}).get("median_s"))
            and finite_positive(
                row.get("integrated_new_frame_first_update_timing", {}).get(
                    "median_s"
                )
            )
            for row in scale_cases.values()
        ),
    }

    direct_c64 = data["odt_direct_c64"]
    direct_c128 = data["odt_direct_c128"]
    matched = direct_c128.get("matched_error_selection", {})
    direct_gates = {
        "c64_audit_passed": bool(direct_c64.get("passed"))
        and all(direct_c64.get("gates", {}).values()),
        "c128_audit_passed": bool(direct_c128.get("passed"))
        and all(direct_c128.get("gates", {}).values()),
        "literal_direct_dot_le_1e_12": direct_c128.get("direct_reference", {}).get(
            "dot_error", float("inf")
        )
        <= 1e-12,
        "acfo_worst_direct_l2_le_2e_6": direct_c128.get("acfo", {}).get(
            "worst_rel_l2_vs_direct", float("inf")
        )
        <= 2e-6,
        "c128_strict_directional_match": direct_c128.get("device", {}).get(
            "cufinufft_dtype"
        )
        == "complex128"
        and bool(matched.get("strict_directional_match_exists"))
        and matched.get("eps") is not None,
    }

    same = data["odt_same_dtype"].get("summary", {})
    same_protocol = same.get("pair_timing_protocol", {})
    same_gates = {
        "same_dtype_complex64_eps_1e_6": same.get("dtype") == "complex64"
        and same.get("eps") == 1e-6,
        "warmup_5_repeat_30_abba": same_protocol.get("method_order")
        == "alternating_ab_ba"
        and same_protocol.get("warmups_per_backend") == 5
        and same_protocol.get("measured_repeats_per_backend") == 30
        and same_protocol.get("ours_distribution", {}).get("count") == 30
        and same_protocol.get("cufinufft_distribution", {}).get("count") == 30,
        "same_dtype_speedup_ge_3": same.get(
            "ours_speedup_vs_cufinufft_pair", 0.0
        )
        >= 3.0,
        "same_dtype_pair_times_finite": finite_positive(
            same.get("ours_forward_adjoint_pair_s")
        )
        and finite_positive(same.get("cufinufft_forward_adjoint_pair_s")),
    }

    matched_full = data["odt_matched_full"]
    matched_cu = matched_full.get("cufinufft", {})
    matched_full_gates = {
        "complex128_eps_1e_7": matched_cu.get("dtype") == "complex128"
        and matched_cu.get("eps") == 1e-7,
        "warmup_2_pair_count_5": matched_cu.get("pair_timing", {}).get("count") == 5,
        "matched_speedup_ge_3": matched_full.get(
            "speedup_acfo_vs_matched_cufinufft", 0.0
        )
        >= 3.0,
        "matched_pair_finite": finite_positive(matched_cu.get("pair_median_s")),
        "external_accuracy_audit_hash": matched_full.get("accuracy_audit", {}).get(
            "sha256"
        )
        == sha256(paths["odt_direct_c128"]),
    }

    temporal = data["odt_temporal"]
    temporal_gates = {
        "schema": temporal.get("schema")
        == "odt-banded-cartesian-temporal-warm-start-v1",
        "all_frozen_sequence_gates": bool(temporal.get("passed"))
        and all(temporal.get("gates", {}).values()),
    }

    numerical_groups = [
        quick_gates,
        {key: value for key, value in prepared_gates.items() if "median_ge" not in key and "p05_ge" not in key},
        {key: value for key, value in detector_gates.items() if "speedup" not in key and "memory_ratio" not in key},
        scale_gates,
        direct_gates,
        {key: value for key, value in same_gates.items() if "speedup" not in key},
        {key: value for key, value in matched_full_gates.items() if "speedup" not in key},
        temporal_gates,
    ]
    performance_gates = {
        "waxs_prepared_speed": prepared_gates["paired_median_ge_3"]
        and prepared_gates["paired_p05_ge_3"],
        "waxs_detector_speed": detector_gates["speedup_gt_1"],
        "waxs_detector_memory": detector_gates[
            "whole_process_memory_ratio_ge_4"
        ],
        "odt_same_dtype_speed": same_gates["same_dtype_speedup_ge_3"],
        "odt_matched_error_speed": matched_full_gates["matched_speedup_ge_3"],
        "odt_temporal_10hz_and_tracking": temporal_gates[
            "all_frozen_sequence_gates"
        ],
    }
    result.update(
        {
            "machine": {
                "reference_fingerprint": local_env.get(
                    "machine_fingerprint_sha256"
                ),
                "measured_fingerprint": environment.get(
                    "machine_fingerprint_sha256"
                ),
                "fingerprints_differ": fingerprints_differ,
                "allow_reference_machine": allow_reference_machine,
            },
            "waxs_prepared_gates": prepared_gates,
            "waxs_detector_gates": detector_gates,
            "odt_scale_gates": scale_gates,
            "odt_direct_gates": direct_gates,
            "odt_same_dtype_gates": same_gates,
            "odt_matched_full_gates": matched_full_gates,
            "odt_temporal_gates": temporal_gates,
            "performance_gates": performance_gates,
            "functional_correctness_pass": all(
                value for group in numerical_groups for value in group.values()
            ),
            "performance_replication_pass": all(performance_gates.values()),
            "independent_machine_replication_pass": fingerprints_differ,
        }
    )
    result["publication_replication_pass"] = bool(
        result["functional_correctness_pass"]
        and result["performance_replication_pass"]
        and result["independent_machine_replication_pass"]
    )
    result["execution_pass"] = bool(
        result["functional_correctness_pass"]
        and result["performance_replication_pass"]
        and (fingerprints_differ or allow_reference_machine)
    )
    result["passed"] = result["execution_pass"]
    result["claim_boundary"] = [
        "publication_replication_pass additionally requires a machine fingerprint different from the local RTX 2070 SUPER reference.",
        "allow_reference_machine can validate the full runner locally, but never converts that run into independent replication.",
        "ODT matched complex128 timing is a separate-process comparison on memory-limited GPUs; the receipt must preserve that caveat.",
        "GPU-resident hot timings exclude acquisition, host transfer and hologram demodulation exactly as in the frozen local protocol.",
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen correctness, timing, independence and evidence gates to an ACFO NCS v14 external run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--allow-reference-machine", action="store_true")
    parser.add_argument("--local-environment", type=Path, default=LOCAL_ENVIRONMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    local_environment = (
        args.local_environment
        if args.local_environment.is_absolute()
        else ROOT / args.local_environment
    )
    result = evaluate_run(
        run_dir,
        mode=args.mode,
        allow_reference_machine=args.allow_reference_machine,
        local_environment=local_environment,
    )
    output = args.output or run_dir / "validation.json"
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    if not result.get("execution_pass", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
