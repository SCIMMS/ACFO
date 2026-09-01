from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/ACFO_NCS_validation_release_candidate_2026-07-13_v13.zip"
RECEIPT = ROOT / "benchmark_results/acfo_ncs_release_candidate_manifest.json"
PREFIX = "ACFO_NCS_validation_release_candidate"


ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    ".gitattributes",
    ".gitignore",
]

STRUCTURES = [
    "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.json",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.json",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz",
    "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.json",
]

WATER_SOURCES = [
    "runs/water_tip3p_8nm/water_tip3p_8nm_final.npz",
    "runs/water_tip3p_8nm/water_tip3p_8nm_trajectory.dcd",
]

RESULT_FILES = [
    "benchmark_results/acfo_ncs_reduced_release_suite.json",
    "benchmark_results/acfo_ncs_fresh_dependency_rerun.json",
    "benchmark_results/local_prepared_waxs_machine_environment.json",
    "benchmark_results/high_na_harmonic_support_risk.json",
    "benchmark_results/waxs_detector_aware_decision.json",
    "benchmark_results/waxs_curvature_isolated_decision.json",
    "benchmark_results/waxs_direct_finufft_triad.json",
    "benchmark_results/waxs_direct_reference_sweep.json",
    "benchmark_results/waxs_direct_reference_sweep.md",
    "benchmark_results/waxs_source_discretization_convergence.json",
    "benchmark_results/waxs_source_discretization_convergence.md",
    "benchmark_results/waxs_exact_beta_harmonic_bridge.json",
    "benchmark_results/waxs_exact_beta_harmonic_bridge.md",
    "benchmark_results/protein_nanocrystal_lattice_factorization.json",
    "benchmark_results/protein_nanocrystal_lattice_factorization.md",
    "benchmark_results/protein_lattice_finufft_512.json",
    "benchmark_results/protein_lattice_finufft_512.md",
    "benchmark_results/protein_lattice_finufft_512_3x3.json",
    "benchmark_results/protein_lattice_finufft_512_5x5.json",
    "benchmark_results/protein_lattice_finufft_512_abba.json",
    "benchmark_results/protein_lattice_finufft_512_abba.md",
    "benchmark_results/protein_lattice_prepared_finufft_512_abba.json",
    "benchmark_results/protein_lattice_prepared_finufft_512_abba.md",
    "benchmark_results/protein_lattice_prepared_abba_decision.json",
    "benchmark_results/protein_lattice_prepared_abba_decision.md",
    "benchmark_results/protein_lattice_q_sampling_decision.json",
    "benchmark_results/protein_lattice_q_sampling_decision.md",
    "benchmark_results/protein_lattice_q_sampling_range_q2p13_chunked.json",
    "benchmark_results/protein_lattice_q_sampling_range_q4p06_chunked.json",
    "benchmark_results/protein_lattice_q_sampling_range_q6p30_chunked.json",
    "benchmark_results/protein_lattice_q_sampling_range_q8p06_chunked.json",
    "benchmark_results/protein_lattice_q_sampling_range_q6p30_reusable_timeout.json",
    "benchmark_results/protein_lattice_q_sampling_resolution_nq32.json",
    "benchmark_results/protein_lattice_q_sampling_resolution_nq128.json",
    "benchmark_results/protein_lattice_q_sampling_resolution_nq512_optimized_profile.json",
    "benchmark_results/protein_lattice_q_sampling_resolution_nq512_fused_phase_profile.json",
    "benchmark_results/exact_beta_contraction_backends_nq512.json",
    "benchmark_results/exact_beta_contraction_backends_nq512.md",
    "benchmark_results/exact_beta_contraction_optimization_decision.json",
    "benchmark_results/exact_beta_contraction_optimization_decision.md",
    "benchmark_results/protein_lattice_highq_threshold_strategy.json",
    "benchmark_results/protein_lattice_highq_threshold_strategy.md",
    "benchmark_results/protein_lattice_highq_threshold_strategy.png",
    "benchmark_results/protein_lattice_highq_threshold_qblock1_nq8_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_qblock2_nq8_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_qblock4_nq8_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_window1_nq16.json",
    "benchmark_results/protein_lattice_highq_threshold_window2_nq16.json",
    "benchmark_results/protein_lattice_highq_threshold_window3_nq16.json",
    "benchmark_results/protein_lattice_highq_threshold_window4_nq16.json",
    "benchmark_results/protein_lattice_highq_threshold_window5_nq16.json",
    "benchmark_results/protein_lattice_highq_threshold_resolution_nq32_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_resolution_nq64_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_resolution_nq128_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_resolution_nq256_q6p70_8p00.json",
    "benchmark_results/protein_lattice_highq_threshold_resolution_nq512_q6p70_8p00.json",
    "benchmark_results/protein_nanocrystal_sparse_accuracy_1iee_1x1x1_highq.json",
    "benchmark_results/tip3p_dense_highq_exact_beta_20frames.json",
    "benchmark_results/tip3p_dense_highq_exact_beta_20frames.md",
    "benchmark_results/tip3p_exact_beta_backend_comparison.json",
    "benchmark_results/tip3p_exact_beta_backend_comparison.md",
    "benchmark_results/tip3p_exact_beta_factorized_q_scaling.json",
    "benchmark_results/tip3p_exact_beta_factorized_q_scaling.md",
    "benchmark_results/tip3p_exact_beta_finufft_512.json",
    "benchmark_results/tip3p_exact_beta_finufft_512.md",
    "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq256_repeat2.json",
    "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq512_w10_r30_alternating.json",
    "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq1024_first.json",
    "benchmark_results/odt_128cubed_gate_decision.json",
    "benchmark_results/odt_256cubed_streaming_decision.json",
    "benchmark_results/odt_streaming_256cubed_first.json",
    "benchmark_results/odt_streaming_256cubed_r16_i4_repeat2.json",
    "benchmark_results/odt_torch_256cubed_100pair.json",
    "benchmark_results/odt_cufinufft_256cubed_100pair.json",
    "benchmark_results/odt_cufinufft_gpu_256cubed_plan_pair2.json",
    "benchmark_results/odt_128cubed_beads_30db_gradient_sweep.json",
    "benchmark_results/odt_128cubed_beads_30db_nonnegative_fista.json",
    "benchmark_results/odt_128cubed_beads_30db_grad_1e6.json",
    "benchmark_results/odt_128cubed_beads_30db_grad_1e7.json",
    "benchmark_results/odt_128cubed_beads_30db_grad_1e8.json",
    "benchmark_results/odt_128cubed_beads_30db_grad_1e9.json",
    "benchmark_results/odt_128cubed_beads_30db_grad_1e10.json",
    "benchmark_results/odt_128cubed_beads_30db_angle60_grad1e8.json",
    "benchmark_results/odt_128cubed_beads_30db_angle70_grad1e8.json",
    "benchmark_results/odt_128cubed_beads_30db_angle80_grad1e8.json",
    "benchmark_results/uniaxial_vector_born_direct_64cubed.json",
    "benchmark_results/uniaxial_green_tensor_residue_64cubed.json",
    "benchmark_results/uniaxial_meep_dispersion_highres_decision.json",
    "benchmark_results/uniaxial_meep_3d_phase_gate_decision.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_gate_decision.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_decision.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_decision.md",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_c10_r12_h025.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_c10_r12_h025.npz",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_c10_r16_h025.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_c10_r16_h025.npz",
    "benchmark_results/uniaxial_meep_component_aware_amplitude_decision.json",
    "benchmark_results/uniaxial_meep_component_aware_amplitude_decision.md",
    "benchmark_results/pymeep_yee_sources_nonlinear_c10_r12_h025.json",
    "benchmark_results/pymeep_yee_sources_nonlinear_c10_r12_h025.npz",
    "benchmark_results/pymeep_yee_sources_nonlinear_c10_r16_h025.json",
    "benchmark_results/pymeep_yee_sources_nonlinear_c10_r16_h025.npz",
    "benchmark_results/pymeep_yee_sources_singlex_c10_r16_h025.json",
    "benchmark_results/pymeep_yee_sources_singlex_c10_r16_h025.npz",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r8_h025.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r8_h025.npz",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r12_h025.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r12_h025.npz",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r16_h025.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c10_r16_h025.npz",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c14_r12_h08.json",
    "benchmark_results/uniaxial_meep_3d_amplitude_singlex_phi45_c14_r12_h08.npz",
]

DOC_FILES = [
    "docs/ACFO_NCS_validation_execution_update_ko.pdf",
    "docs/acfo_ncs_validation_execution_update_source_notes.md",
    "docs/acfo_ncs_high_na_harmonic_cutoff_ko.md",
    "docs/acfo_ncs_waxs_detector_aware_ko.md",
    "docs/acfo_ncs_waxs_direct_reference_ko.md",
    "docs/acfo_ncs_odt_256cubed_streaming_ko.md",
    "docs/acfo_ncs_odt_operator_vs_missing_cone_ko.md",
    "docs/acfo_ncs_odt_regularized_inverse_ko.md",
    "docs/acfo_ncs_anisotropic_green_reference_ko.md",
    "docs/acfo_ncs_external_rerun_handoff_ko.md",
    "docs/acfo_ncs_step5_fullwave_backend_audit_ko.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    paths = [ROOT / item for item in ROOT_FILES + STRUCTURES + WATER_SOURCES + RESULT_FILES + DOC_FILES]
    paths.extend(sorted((ROOT / "src/waxs_cake").glob("**/*")))
    paths.extend(sorted((ROOT / "tests").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*.ps1")))
    return [path for path in paths if path.is_file() and "__pycache__" not in path.parts]


def main() -> None:
    files = selected_files()
    missing = [item for item in ROOT_FILES + STRUCTURES + WATER_SOURCES + RESULT_FILES + DOC_FILES if not (ROOT / item).exists()]
    if missing:
        raise RuntimeError(f"missing required release files: {missing}")
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    release_readme = r"""# ACFO NCS validation release candidate

## Scope

This package reproduces the local reduced validation suite and carries the saved production-scale evidence used by the accompanying PDF. It is a release candidate, not an independent-machine validation receipt.

## Environment setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .[reproduce]
```

Install the CUDA-compatible PyTorch build appropriate for the external GPU if the generic `gpu` extra does not select it.
Install `.[gpu-baseline]` as well when rerunning the cuFINUFFT production comparison.

## One-command reduced suite

```powershell
.\.venv\Scripts\python.exe scripts\run_acfo_ncs_reduced_release_suite.py
```

Expected local reference: full pytest, High-NA charge sweep, detector-mask smoke, WAXS direct-NDFT q/curvature/source-discretization, fused exact-beta and perfect/sparse-defect lattice controls, 64-cubed compact ODT streaming smoke, production-artifact schema checks, prepared-ABBA/q-sampling summaries, high-q chart rebuild and PDF rebuild in under five minutes. The saved TIP3P 20-frame, q-scaling, Nq512 FINUFFT, legacy/prepared 1M AB/BA, q-range/resolution and high-q threshold comparisons are schema-checked rather than rerun by the reduced suite.

## One-command independent prepared-WAXS timing

After the editable install and forced C++ extension rebuild, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_external_prepared_waxs_validation.ps1
```

This records the machine and source fingerprints, runs the reduced suite, performs the 1.001M-atom prepared fused 10/30 AB/BA comparison, applies the independent-machine and numerical gates, and creates `benchmark_results/external_prepared_waxs_return_package.zip`. Use `-Resume` after interruption. The runner deliberately rejects a result carrying the reference machine fingerprint.

## Fresh-dependency receipt

`benchmark_results/acfo_ncs_fresh_dependency_rerun.json` records the earlier isolated-venv dependency rebuild on the reference machine: four C++ extensions rebuilt, CUDA-enabled PyTorch detected, and the then-current 172 tests passed. The final archive clean-source suite now contains 189 tests and is the current code receipt. The reduced suite selects CUDA for its compact ODT smoke when available and otherwise runs the same smoke on CPU. Neither run is an independent-machine replication. The optional cuFINUFFT extra was not installed in that isolated environment; the saved production comparison remains the applicable cuFINUFFT evidence.

## Production evidence versus rerun scope

| Item | Included evidence | Reduced suite reruns it? | External action |
|---|---|---:|---|
| High-NA charge 0-24 | JSON + script | yes | repeat on an independent machine |
| WAXS direct/exact-beta/lattice controls | 1x1x1, 3x3x3, 5x5x5 structures + legacy/prepared local 10/30 AB/BA JSON | controls yes; timing schema only | repeat the prepared 1M 10/30 protocol on an independent machine |
| WAXS q-range/resolution and contraction | fixed-dq chunked, fixed-range reusable, fused-path profile, prepared 10/30 and backend A/B JSON | schema only | detector-realistic independent repeat |
| WAXS high-q threshold strategy | q-block calibration, five equal-width windows, Nq32 completion, Nq64 censored lower bound and holdout-gated projection | schema + chart only | independently repeat measured rows; keep projection secondary |
| TIP3P dense exact-coordinate path | DCD, final NPZ, 20-frame and Nq512 JSON | schema only | reproduce full saved benchmarks independently |
| WAXS 216k Nq256/512/1024 | JSON + 3x3x3 structure | no | rerun Nq512 10/30 AB/BA protocol on an external machine |
| ODT 64-cubed streaming | synthetic generator | yes | verify CUDA path |
| ODT 256-cubed streaming and 100 pairs | ACFO/cuFINUFFT JSON + scripts | no | independently rerun both standalone workflows |
| PyMeep Maxwell results | saved decision JSON | no | WSL/PyMeep environment required |

## Claim boundary

Missing-cone beads recovery is an acquisition/prior robustness question, not an ACFO forward/adjoint correctness gate. WAXS full-harmonic and exact-beta paths pass direct NDFT at 1e-12 or better; perfect 1M lattice and 0.1% sparse-defect delta controls pass at 2.43e-11. These are standard crystallographic specializations, not ACFO novelty. The current 0.1 nm/Nphi 750 coarse high-q representation still fails, while the fused exact-coordinate path passes 20 TIP3P frames at max complex L2 2.46e-12. In dense Nq512, reusable FINUFFT is about 40x faster first-total and 99x faster hot. In exact repeated crystals, 216k remains FINUFFT-favorable. The 1.001M-atom prepared fused same-machine 10-warmup/30-pair AB/BA run gives factorized/FINUFFT medians 3.143/106.945 s, paired median 33.480x, p05 24.243x and AB/BA median gap 2.04%. The prepared factorized median is 4.427x faster than the legacy receipt at complex L2 4.81e-14. The split sweep reproduces the earlier fixed-dq behavior: qmax 2.13 to 8.06 A^-1 increases q-blocked first-total speedup from 47.0x to 147.8x, while fixed-range Nq 32 to 512 reduces reusable hot speedup from 314.3x to 33.5x. In the equal-width q-window sweep, measured q-blocked speedup increases from 79.2x to 487.8x. At q=6.7-8.0 A^-1, Nq32 completes at 538.2x and Nq64 stops after 42/64 rows at 181.7 s, giving a measured >343.0x lower bound. Holdout-gated extrapolations remain secondary planning estimates. An exact phase cache adds only 1.169x median hot speedup while consuming 42.89 MiB, so it remains optional. The local prepared timing gate passes; independent-machine replication remains required. ODT streaming feasibility passes, but the standalone 100-pair speed gate fails. Exact Yee-source nonlinear PyMeep is a detectable-support scoped PASS while publication full amplitude remains FAIL.
"""
    manifest = {
        "schema": "acfo-ncs-validation-release-candidate-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_prefix": PREFIX,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "file_count": len(entries),
        "payload_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "limitations": [
            "A same-machine fresh-dependency receipt is included; this packaging command does not rerun that workflow.",
            "Independent-machine rerun remains external work.",
            "Production WAXS and ODT rows are included as immutable evidence and are not rerun by the reduced suite.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(f"{PREFIX}/README_RELEASE.md", release_readme)
        archive.writestr(
            f"{PREFIX}/MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        for path in files:
            archive.write(path, f"{PREFIX}/{path.relative_to(ROOT).as_posix()}")
    receipt = {
        **manifest,
        "zip": OUTPUT.relative_to(ROOT).as_posix(),
        "zip_bytes": OUTPUT.stat().st_size,
        "zip_sha256": sha256(OUTPUT),
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT} and {RECEIPT}")


if __name__ == "__main__":
    main()
