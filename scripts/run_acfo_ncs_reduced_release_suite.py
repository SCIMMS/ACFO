from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "acfo_ncs_reduced_release_suite.json"


def choose_smoke_device(torch_module) -> str:
    return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"


def run(label: str, args: list[str], *, timeout: int) -> dict:
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    record = {
        "label": label,
        "command": args,
        "returncode": completed.returncode,
        "duration_s": elapsed,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
        "passed": completed.returncode == 0,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def require_json(name: str, schema: str, *, passed_key: str = "passed") -> dict:
    path = RESULTS / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise RuntimeError(f"{name}: expected schema {schema}, got {payload.get('schema')}")
    if not bool(payload.get(passed_key)):
        raise RuntimeError(f"{name}: {passed_key} is not true")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "schema": schema,
        "passed_key": passed_key,
        "passed": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    import torch

    py = sys.executable
    smoke_device = choose_smoke_device(torch)
    commands = []
    suite_start = time.perf_counter()
    commands.append(run("full_pytest", [py, "-m", "pytest", "-q"], timeout=300))
    commands.append(
        run(
            "high_na_charge_sweep",
            [py, "scripts/validate_high_na_harmonic_support_risk.py"],
            timeout=180,
        )
    )
    commands.append(
        run(
            "detector_mask_smoke",
            [
                py,
                "scripts/benchmark_protein_nanocrystal_finufft_fair.py",
                "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
                "--source-mode", "same_binned", "--qmin", "0.05", "--qmax", "6.3",
                "--nq", "8", "--wavelength-nm", "0.08", "--bin-width-nm", "0.1",
                "--nphi-min", "128", "--harmonic-margin", "32", "--r-dependent-margin", "32",
                "--cutoff-bin-size", "16", "--acfo-q-block-size", "2", "--profile-chunk-size", "8",
                "--finufft-eps", "1e-6", "--finufft-threads", "4", "--finufft-q-block-size", "2",
                "--repeats", "0", "--detector-label", "EIGER2_X_4M_15p5keV_100mm",
                "--detector-active-width-mm", "155.1", "--detector-active-height-mm", "162.15",
                "--detector-distance-mm", "100",
                "--output", "benchmark_results/protein_nanocrystal_detector_mask_smoke.json",
            ],
            timeout=180,
        )
    )
    commands.append(
        run(
            "waxs_direct_finufft_triad",
            [
                py,
                "scripts/validate_waxs_direct_finufft_triad.py",
                "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
                "--output",
                "benchmark_results/waxs_direct_finufft_triad.json",
            ],
            timeout=180,
        )
    )
    commands.append(
        run(
            "waxs_direct_reference_sweep",
            [
                py,
                "scripts/validate_waxs_direct_reference_sweep.py",
                "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
            ],
            timeout=300,
        )
    )
    commands.append(
        run(
            "waxs_source_discretization_convergence",
            [
                py,
                "scripts/validate_waxs_source_discretization_convergence.py",
                "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
            ],
            timeout=180,
        )
    )
    commands.append(
        run(
            "waxs_exact_beta_harmonic_bridge",
            [
                py,
                "scripts/validate_waxs_exact_beta_harmonic_bridge.py",
                "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
                "--bessel-backend",
                "cpp_fused",
            ],
            timeout=180,
        )
    )
    commands.append(
        run(
            "protein_nanocrystal_lattice_factorization",
            [py, "scripts/validate_protein_nanocrystal_lattice_factorization.py"],
            timeout=300,
        )
    )
    commands.append(
        run(
            "odt_compact_streaming_64cubed_smoke",
            [
                py,
                "scripts/benchmark_odt_torch_gpu_reconstruction.py",
                "--device", smoke_device, "--dtype", "complex64", "--low-memory-adjoint",
                "--radial-block-size", "16", "--illumination-block-size", "4",
                "--skip-native-prepared-adjoint", "--compact-axisymmetric-kernel", "--real-object",
                "--n-beta", "64", "--n-r", "64", "--n-z", "64", "--ring-illum", "12",
                "--cap-radial", "64", "--cap-phi", "64", "--h-margin", "8", "--l-margin", "8",
                "--cone-l-prune-threshold", "0", "--cpp-threads", "4", "--iterations", "1",
                "--repeats", "1", "--warmups", "0",
                "--out", "benchmark_results/odt_streaming_64cubed_smoke.json",
                "--csv", "benchmark_results/odt_streaming_64cubed_smoke.csv",
                "--summary-md", "benchmark_results/odt_streaming_64cubed_smoke.md",
            ],
            timeout=180,
        )
    )
    for label, script in (
        ("summarize_waxs_detector", "scripts/summarize_waxs_detector_aware.py"),
        ("summarize_odt_streaming", "scripts/summarize_odt_256_streaming.py"),
        ("summarize_odt_inverse", "scripts/summarize_odt_regularized_inverse.py"),
        ("summarize_prepared_abba", "scripts/summarize_protein_lattice_prepared_abba.py"),
        ("summarize_q_sampling", "scripts/summarize_protein_lattice_q_sampling.py"),
        ("plot_highq_threshold", "scripts/plot_protein_lattice_highq_threshold_strategy.py"),
    ):
        commands.append(run(label, [py, script], timeout=180))

    production_artifacts = [
        require_json(
            "local_prepared_waxs_machine_environment.json",
            "prepared-waxs-machine-environment-v1",
        ),
        require_json("high_na_harmonic_support_risk.json", "high-na-harmonic-support-risk-v2"),
        require_json("waxs_detector_aware_decision.json", "waxs-detector-aware-decision-v2"),
        require_json("waxs_direct_finufft_triad.json", "waxs-direct-finufft-triad-v1"),
        require_json(
            "waxs_direct_reference_sweep.json",
            "waxs-direct-reference-sweep-v1",
            passed_key="operator_reference_pass",
        ),
        require_json(
            "waxs_source_discretization_convergence.json",
            "waxs-source-discretization-convergence-v1",
        ),
        require_json(
            "waxs_exact_beta_harmonic_bridge.json",
            "waxs-exact-beta-harmonic-bridge-v1",
        ),
        require_json(
            "protein_nanocrystal_lattice_factorization.json",
            "protein-nanocrystal-lattice-factorization-v1",
        ),
        require_json(
            "protein_lattice_finufft_512.json",
            "protein-lattice-finufft-512-crossover-v1",
        ),
        require_json(
            "protein_lattice_finufft_512_abba.json",
            "protein-lattice-finufft-512-abba-v1",
        ),
        require_json(
            "protein_lattice_prepared_finufft_512_abba.json",
            "protein-lattice-prepared-finufft-512-abba-v1",
        ),
        require_json(
            "protein_lattice_prepared_abba_decision.json",
            "protein-lattice-prepared-abba-decision-v1",
        ),
        require_json(
            "protein_lattice_q_sampling_decision.json",
            "protein-lattice-q-sampling-decision-v1",
        ),
        require_json(
            "exact_beta_contraction_optimization_decision.json",
            "exact-beta-contraction-optimization-decision-v1",
        ),
        require_json(
            "protein_lattice_highq_threshold_strategy.json",
            "protein-lattice-highq-threshold-strategy-v1",
        ),
        require_json(
            "tip3p_dense_highq_exact_beta_20frames.json",
            "tip3p-dense-highq-exact-beta-v2",
        ),
        require_json(
            "tip3p_exact_beta_backend_comparison.json",
            "tip3p-exact-beta-backend-comparison-v1",
        ),
        require_json(
            "tip3p_exact_beta_factorized_q_scaling.json",
            "tip3p-exact-beta-factorized-q-scaling-v1",
        ),
        require_json(
            "tip3p_exact_beta_finufft_512.json",
            "tip3p-exact-beta-finufft-512-v1",
        ),
        require_json(
            "uniaxial_green_tensor_residue_64cubed.json",
            "uniaxial-green-tensor-residue-v1",
        ),
        require_json(
            "uniaxial_meep_3d_amplitude_asymptotic_decision.json",
            "uniaxial-meep-3d-amplitude-asymptotic-decision-v1",
            passed_key="scoped_single_component_amplitude_pass",
        ),
        require_json(
            "uniaxial_meep_component_aware_amplitude_decision.json",
            "uniaxial-meep-component-aware-amplitude-decision-v1",
            passed_key="detectable_support_amplitude_pass",
        ),
        require_json(
            "odt_256cubed_streaming_decision.json",
            "odt-256cubed-streaming-decision-v2",
            passed_key="streaming_feasibility_pass",
        ),
        require_json(
            "odt_128cubed_gate_decision.json",
            "acfo-odt-128cubed-gate-v1",
            passed_key="minimum_odt_128cubed_numerical_gate_pass",
        ),
    ]
    result = {
        "schema": "acfo-ncs-reduced-release-suite-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": time.perf_counter() - suite_start,
        "commands": commands,
        "production_artifact_manifest": production_artifacts,
        "recorded_outcomes": {
            "odt_256cubed_comparative_performance_pass": json.loads(
                (RESULTS / "odt_256cubed_streaming_decision.json").read_text(encoding="utf-8")
            )["comparative_performance_pass"],
            "pymeep_publication_full_amplitude_pass": json.loads(
                (RESULTS / "uniaxial_meep_component_aware_amplitude_decision.json").read_text(encoding="utf-8")
            )["publication_full_amplitude_pass"],
            "pymeep_grid_converged_detectable_support_pass": json.loads(
                (RESULTS / "uniaxial_meep_component_aware_amplitude_decision.json").read_text(encoding="utf-8")
            )["grid_converged_detectable_support_pass"],
            "waxs_current_production_source_representation_pass": False,
            "waxs_exact_beta_unit_cell_pass": True,
            "waxs_perfect_and_sparse_defect_lattice_control_pass": True,
            "waxs_dense_disorder_general_source_path_pass": True,
            "waxs_dense_disorder_comparative_performance_pass": False,
            "waxs_dense_corrected_nq512_timing_measured": True,
            "waxs_million_atom_repeated_crystal_local_10_30_pass": True,
            "waxs_prepared_fused_million_atom_local_10_30_pass": True,
            "waxs_prepared_fused_independent_machine_pass": False,
            "waxs_fixed_dq_q_range_speedup_monotonic_pass": True,
            "waxs_fixed_range_resolution_trend_pass": True,
            "waxs_prepared_exact_optimization_probe_pass": True,
            "waxs_exact_beta_fused_phase_contraction_pass": True,
            "waxs_highq_equal_width_position_sweep_pass": True,
            "waxs_highq_nq64_censored_lower_bound_pass": True,
            "waxs_highq_extrapolation_holdout_gate_pass": True,
        },
        "pdf": {"status": "pending"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "odt_smoke_device": smoke_device,
        },
        "scope": (
            "Local reduced reproduction: full unit tests, direct-reference High-NA sweep, detector-mask smoke, "
            "small WAXS direct-NDFT/FINUFFT/ACFO triad, q/curvature/source-discretization sweeps, exact-beta and crystal-lattice controls, "
            "saved TIP3P 20-frame dense correctness, backend, Nq-scaling, matched FINUFFT, legacy and prepared fused 1M lattice 10/30 AB/BA, q-range/resolution, high-q censored-threshold and fused-contraction evidence, "
            "64-cubed compact-streaming ODT smoke, production-artifact schema gates, summaries and PDF rebuild."
        ),
        "limitations": [
            "The 216k WAXS and 256-cubed ODT production benchmarks are schema-verified saved artifacts, not rerun here.",
            "The TIP3P 20-frame, Nq=512 dense, legacy/prepared 1M repeated-crystal 10/30, q-range/resolution, high-q threshold and contraction A/B benchmarks are schema-verified saved artifacts, not rerun by this reduced suite.",
            "This is not a fresh-install or independent-machine rerun.",
        ],
        "passed": True,
    }
    # Write a current receipt before the PDF build so a fresh workspace has no
    # circular dependency: the report can cite this same suite run.
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        commands.append(
            run(
                "build_execution_update_pdf",
                [py, "scripts/build_acfo_ncs_execution_update_pdf.py"],
                timeout=180,
            )
        )
    except Exception:
        result["passed"] = False
        result["pdf"] = {"status": "failed"}
        result["commands"] = commands
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    pdf = ROOT / "docs/ACFO_NCS_validation_execution_update_ko.pdf"
    if not pdf.exists() or pdf.stat().st_size < 100_000:
        raise RuntimeError("execution-update PDF is missing or unexpectedly small")
    result["duration_s"] = time.perf_counter() - suite_start
    result["commands"] = commands
    result["pdf"] = {
        "status": "passed",
        "path": pdf.relative_to(ROOT).as_posix(),
        "size_bytes": pdf.stat().st_size,
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "commands"}, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
