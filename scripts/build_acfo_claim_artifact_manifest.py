from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "benchmark_results/acfo_claim_artifact_manifest.json"
OUTPUT_MD = ROOT / "docs/acfo_claim_artifact_manifest_ko.md"
RELEASE_MANIFEST = ROOT / "benchmark_results/acfo_ncs_release_candidate_manifest.json"


def read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {relative_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    try:
        status_lines = [
            line for line in run("status", "--porcelain=v1").splitlines() if line
        ]
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest_path = ROOT / "MANIFEST.json"
        release = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        revision = release.get("source_revision", {})
        return {
            "commit": revision.get("commit", "packaged-source"),
            "branch": revision.get("branch", "release-archive"),
            "tracked_changes": revision.get("tracked_changes"),
            "untracked_entries": revision.get("untracked_entries"),
            "total_status_entries": revision.get("total_status_entries"),
            "clean": revision.get("clean"),
            "source": "embedded release manifest; .git unavailable",
        }
    return {
        "commit": commit,
        "branch": branch,
        "tracked_changes": sum(not line.startswith("??") for line in status_lines),
        "untracked_entries": sum(line.startswith("??") for line in status_lines),
        "total_status_entries": len(status_lines),
        "clean": not status_lines,
    }


def release_hashes() -> tuple[dict[str, str], dict[str, Any]]:
    release = read_json(RELEASE_MANIFEST.relative_to(ROOT).as_posix())
    hashes = {entry["path"]: entry["sha256"] for entry in release["files"]}
    metadata = {
        "path": RELEASE_MANIFEST.relative_to(ROOT).as_posix(),
        "generated_at_utc": release["generated_at_utc"],
        "zip": release["zip"],
        "zip_sha256": release["zip_sha256"],
        "file_count": release["file_count"],
    }
    return hashes, metadata


def artifact(
    relative_path: str,
    role: str,
    release_entries: dict[str, str],
) -> dict[str, Any]:
    normalized = Path(relative_path).as_posix()
    path = ROOT / normalized
    if not path.is_file():
        raise FileNotFoundError(f"required claim artifact is missing: {normalized}")
    current_hash = sha256(path)
    release_hash = release_entries.get(normalized)
    return {
        "path": normalized,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": current_hash,
        "release_candidate": {
            "included": release_hash is not None,
            "hash_matches_current": release_hash == current_hash if release_hash else None,
        },
    }


def claim_artifacts(
    entries: list[tuple[str, str]], release_entries: dict[str, str]
) -> list[dict[str, Any]]:
    return [artifact(path, role, release_entries) for path, role in entries]


def build_manifest() -> dict[str, Any]:
    release_entries, release_metadata = release_hashes()

    waxs_detector = read_json("benchmark_results/waxs_detector_aware_decision.json")
    waxs_512 = next(row for row in waxs_detector["rows"] if row["nq"] == 512)
    require(waxs_detector["passed"], "WAXS detector-aware decision must pass")
    require(
        waxs_512["timing_protocol"]["warmups_per_method"] == 10
        and waxs_512["timing_protocol"]["measured_repeats_per_method"] == 30,
        "WAXS Nq=512 timing protocol is not the frozen 10/30 protocol",
    )

    tip3p = read_json("benchmark_results/tip3p_dense_highq_exact_beta_20frames.json")
    tip3p_finufft = read_json("benchmark_results/tip3p_exact_beta_finufft_512.json")
    require(tip3p["passed"] and tip3p["frame_count"] == 20, "TIP3P exact-beta gate failed")
    require(
        not tip3p_finufft["comparative_performance_pass"],
        "TIP3P Nq=512 comparison unexpectedly changed status",
    )

    aidt = read_json("benchmark_results/aidt_10hz_full700_opt_repeat.json")
    aidt_summary = aidt["summary"]
    aidt_hz = 1.0 / aidt_summary["gpu_run_median_s"]
    require(aidt_hz >= 10.0, "aIDT GPU-resident core no longer reaches 10 Hz")
    require(
        aidt_summary["processed_shape"] == [24, 700, 700]
        and aidt_summary["depth_count"] == 35,
        "aIDT frozen public condition changed",
    )

    odt_validation = read_json("benchmark_results/odt_final_packed_candidate_validation.json")
    odt_abba = read_json(
        "benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json"
    )
    odt_abba_summary = odt_abba["summary"]
    require(odt_validation["passed"], "ODT final packed candidate accuracy gate failed")
    require(
        odt_abba_summary["pair_timing_protocol"]["measured_repeats_per_backend"] == 30,
        "ODT packed/cuFINUFFT comparison is not the frozen 30-repeat protocol",
    )

    odt_reconstruction = read_json("benchmark_results/odt_full_slab_reconstruction_claim.json")
    full_case = next(case for case in odt_reconstruction["cases"] if case["selected_n_z"] == 256)
    slab128_case = next(
        case for case in odt_reconstruction["cases"] if case["selected_n_z"] == 128
    )
    full_recon = full_case["reconstruction"]
    slab128_recon = slab128_case["reconstruction"]
    full_median_s = full_recon["iteration_core_timing"]["median_s"]
    slab128_median_s = slab128_recon["iteration_core_timing"]["median_s"]

    odt_detector = read_json("benchmark_results/odt_banded_cartesian_reconstruction.json")
    odt_remap_recheck = read_json(
        "benchmark_results/odt_banded_cartesian_remap_hot_recheck.json"
    )
    detector_z1 = next(case for case in odt_detector["cases"] if case["selected_n_z"] == 1)
    detector_z8 = next(case for case in odt_detector["cases"] if case["selected_n_z"] == 8)
    recheck_z1 = odt_remap_recheck["cases"][0]
    z1_total_s = (
        detector_z1["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + recheck_z1["remap_timing"]["median_s"]
    )
    z8_total_s = (
        detector_z8["cartesian_remap"]["core_iteration_timing"]["median_s"]
        + detector_z8["remap_timing"]["median_s"]
    )

    odt_integrated = read_json(
        "benchmark_results/odt_banded_cartesian_final_packed_probe.json"
    )
    odt_integrated_timing = read_json(
        "benchmark_results/odt_banded_cartesian_final_packed_full_timing.json"
    )
    odt_matched_c64 = read_json(
        "benchmark_results/odt_cufinufft_matched_error_direct_subset.json"
    )
    odt_matched_c128 = read_json(
        "benchmark_results/odt_cufinufft_matched_error_direct_subset_c128.json"
    )
    odt_matched_full = read_json(
        "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json"
    )
    odt_temporal = read_json(
        "benchmark_results/odt_banded_cartesian_temporal_warm_start.json"
    )
    waxs_protein_decision = read_json(
        "benchmark_results/waxs_protein_exact_beta_followup_decision.json"
    )
    require(odt_integrated["passed"], "ODT integrated detector/operator gate failed")
    require(
        len(odt_integrated_timing["cases"]) == 3,
        "ODT integrated timing scale-up must contain 64/128/256-z rows",
    )
    require(odt_matched_c64["passed"], "ODT complex64 direct audit failed")
    require(odt_matched_c128["passed"], "ODT complex128 direct audit failed")
    require(
        odt_matched_c128["matched_error_selection"]["eps"] == 1e-7,
        "ODT matched-error cuFINUFFT epsilon changed",
    )
    require(
        odt_matched_full["cufinufft"]["pair_timing"]["count"] == 5,
        "ODT matched-error full timing is not the frozen five-repeat run",
    )
    require(odt_temporal["passed"], "ODT integrated temporal gate failed")
    require(waxs_protein_decision["passed"], "WAXS protein follow-up decision failed")

    curvature = read_json(
        "benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24.json"
    )
    require(curvature["passed"], "general-curvature frozen holdout no longer passes")
    require(
        curvature["holdout_contract"]["no_holdout_refit"],
        "general-curvature holdout was refitted",
    )
    require(all(curvature["gates"].values()), "not all general-curvature gates pass")
    curvature_row = curvature["rows"][0]
    curvature_inner_shell = curvature_row["shells"][0]

    claims = [
        {
            "claim_id": "waxs_detector_aware_local",
            "application": "WAXS",
            "status": "verified_local_external_rerun_pending",
            "headline": "EIGER2 X 4M envelope의 Nq=512 local 10/30 protocol에서 1.976x",
            "claim_ko": "동일 binned source와 detector-aware curved targets에서 prepared ACFO가 FINUFFT보다 약 1.98배 빠르다.",
            "scope": "RTX 2070 SUPER local CPU timing; Nq=512; geometric detector envelope; fixed source representation",
            "reference": "FINUFFT on the same binned source and active curved targets",
            "timing_boundary": "10 warm-ups and 30 AB/BA alternating cached solves; setup excluded",
            "metrics": {
                "acfo_median_s": waxs_512["acfo_cached_s"],
                "finufft_median_s": waxs_512["finufft_cached_s"],
                "ratio_of_medians_speedup": waxs_512["warm_speedup"],
                "paired_p05_speedup": waxs_512["paired_timing"]["p05_speedup"],
                "complex_l2": waxs_512["complex_l2"],
                "memory_reduction_ratio": waxs_512["memory_reduction_ratio"],
            },
            "claim_boundary": [
                "This is same-bin operator evidence, not end-to-end exact-atom high-q accuracy.",
                "Module gaps and beamstop are not included in the geometric-envelope detector tier.",
                "Independent-machine timing remains pending.",
            ],
            "reproduction": {
                "mode": "saved production rows plus deterministic decision rebuild",
                "driver": "scripts/summarize_waxs_detector_aware.py",
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/waxs_detector_aware_decision.json", "decision"),
                    (
                        "benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq512_w10_r30_alternating.json",
                        "raw timing and accuracy",
                    ),
                    ("scripts/summarize_waxs_detector_aware.py", "decision builder"),
                    ("docs/acfo_ncs_waxs_detector_aware_ko.md", "human-readable interpretation"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "waxs_dense_exact_beta_highq",
            "application": "WAXS",
            "status": "accuracy_verified_dense_nq512_not_competitive",
            "headline": "50,430-atom TIP3P 20 frames의 high-q exact-beta 정확도 PASS; dense Nq=512 성능 FAIL",
            "claim_ko": "Bounded-memory exact-beta 경로는 dense disordered coordinates를 direct NDFT 정확도로 재현하지만 dense Nq=512 workload에서는 FINUFFT보다 느리다.",
            "scope": "8 nm TIP3P trajectory; q=5.0-6.3 A^-1; local CPU wall time",
            "reference": "direct NDFT for correctness; reusable FINUFFT for Nq=512 cross-check",
            "timing_boundary": "exact-coordinate contraction and harmonic evaluation; file loading/form-factor construction excluded in the scaling row",
            "metrics": {
                "frames": tip3p["frame_count"],
                "atoms_per_frame": tip3p["atom_count"],
                "max_complex_l2_vs_direct": tip3p["aggregate"]["exact_beta_complex_l2"]["max"],
                "median_direct_over_exact_beta": tip3p["aggregate"][
                    "direct_over_exact_beta_speedup"
                ]["median"],
                "nq512_exact_beta_s": tip3p_finufft["cpp_fused"]["seconds"],
                "nq512_finufft_hot_median_s": tip3p_finufft["finufft"]["hot_seconds"][
                    "median"
                ],
                "nq512_cross_complex_l2": tip3p_finufft["cross_error"]["complex_l2"],
                "nq512_comparative_performance_pass": tip3p_finufft[
                    "comparative_performance_pass"
                ],
            },
            "claim_boundary": tip3p["claim_boundary"] + tip3p_finufft["claim_boundary"],
            "reproduction": {
                "mode": "full saved TIP3P validation driver",
                "driver": "scripts/validate_tip3p_dense_highq_exact_beta.py",
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/tip3p_dense_highq_exact_beta_20frames.json", "accuracy"),
                    ("benchmark_results/tip3p_exact_beta_finufft_512.json", "dense performance"),
                    ("scripts/validate_tip3p_dense_highq_exact_beta.py", "accuracy driver"),
                    ("scripts/benchmark_tip3p_exact_beta_finufft_512.py", "FINUFFT comparison driver"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "aidt_gpu_resident_core",
            "application": "aIDT",
            "status": "verified_processing_core",
            "headline": "24x700x700 input to 700x700x35 GPU-resident core: 97.002 ms, 10.31 Hz",
            "claim_ko": "고정 geometry와 GPU-resident input의 public aIDT reconstruction core가 RTX 2070 SUPER에서 10 Hz를 넘는다.",
            "scope": "public 24-illumination aIDT transfer reconstruction; complex64; prepared support-transfer cache",
            "reference": "validated public-data contract and dense transfer baseline",
            "timing_boundary": "GPU core only; setup, output statistics, CPU preprocessing, acquisition and scheduling excluded",
            "metrics": {
                "gpu_run_median_s": aidt_summary["gpu_run_median_s"],
                "gpu_run_hz": aidt_hz,
                "gpu_setup_s": aidt_summary["gpu_setup_s"],
                "n_illumination": aidt_summary["n_illum"],
                "depth_count": aidt_summary["depth_count"],
                "support_active_fraction": aidt_summary["support_active_fraction"],
                "torch_peak_allocated_mib": aidt_summary["torch_peak_allocated_mib"],
            },
            "claim_boundary": [
                "This is a GPU-resident processing-core result, not raw-acquisition-to-reconstruction end-to-end 10 Hz.",
                "Output statistics were deliberately excluded in the frozen 10 Hz row.",
            ],
            "reproduction": {
                "mode": "full public-condition benchmark using saved JSON config",
                "driver": "scripts/benchmark_aidt_transfer_torch_gpu.py",
                "saved_config": aidt["config"],
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/aidt_10hz_full700_opt_repeat.json", "timing and configuration"),
                    (
                        "validation/odt_aidt/aidt_diatom_public_contract_validation.json",
                        "public-data contract validation receipt",
                    ),
                    ("scripts/benchmark_aidt_transfer_torch_gpu.py", "benchmark driver"),
                    ("benchmark_results/aidt_10hz_full700_opt_repeat.md", "human-readable summary"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "odt_final_packed_operator",
            "application": "ODT",
            "status": "operator_verified_local_timing_baseline_accuracy_open",
            "headline": "256^3 final packed pair 111.366 ms; H36 structured gate PASS; local cuFINUFFT ratio 81.24x",
            "claim_ko": "Final H28/rank16/adaptive-L packed operator는 H36 structured reference gate를 통과하고 local reusable-plan timing에서 큰 실행시간 차이를 보인다.",
            "scope": "256^3; 121 illuminations; 7,929,856 q samples; fixed geometry; complex64",
            "reference": "H36 full-rank dense-L structured operator for ACFO accuracy; cuFINUFFT reusable plan for timing",
            "timing_boundary": "both inputs GPU-resident; setup/allocation/setpts excluded; 5 warm-ups plus 30 AB/BA pairs",
            "metrics": {
                "worst_rel_l2_vs_h36": odt_validation["worst_rel_l2"],
                "acceptance_tolerance": odt_validation["tolerance"],
                "dot_error": odt_validation["dot_error"],
                "acfo_pair_median_s": odt_abba_summary["ours_forward_adjoint_pair_s"],
                "cufinufft_pair_median_s": odt_abba_summary[
                    "cufinufft_forward_adjoint_pair_s"
                ],
                "local_pair_speedup": odt_abba_summary[
                    "ours_speedup_vs_cufinufft_pair"
                ],
                "cross_backend_forward_rel_l2": odt_abba_summary[
                    "cufinufft_forward_rel_l2_vs_ours"
                ],
                "cross_backend_adjoint_rel_l2": odt_abba_summary[
                    "cufinufft_adjoint_rel_l2_vs_ours"
                ],
            },
            "claim_boundary": [
                "The 81.24x value is a measured local hot-time ratio, not yet a matched-effective-error speedup.",
                "The ACFO accuracy gate is against H36 structured ACFO; the cross-backend difference is not an independent exact error.",
                "Independent-machine replication remains pending.",
            ],
            "reproduction": {
                "mode": "accuracy driver plus saved-config production AB/BA driver",
                "accuracy_driver": "scripts/validate_odt_final_packed_candidate.py",
                "timing_driver": "scripts/benchmark_odt_cufinufft_gpu_baseline.py",
                "saved_timing_config": odt_abba["config"],
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/odt_final_packed_candidate_validation.json", "structured accuracy"),
                    (
                        "benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json",
                        "local hot timing",
                    ),
                    ("scripts/validate_odt_final_packed_candidate.py", "accuracy driver"),
                    ("scripts/benchmark_odt_cufinufft_gpu_baseline.py", "timing driver"),
                    ("benchmark_results/odt_promising_optimization_final_summary_ko.md", "decision summary"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "odt_full_slab_update_throughput",
            "application": "ODT",
            "status": "verified_synthetic_update_throughput",
            "headline": "full 256^3 update 8.46 Hz; known-support z=128 update 10.64 Hz",
            "claim_ko": "Final packed operator를 이용한 fixed-geometry normal-equation update가 full volume에서 8.46 Hz, z=128 known-support slab에서 10 Hz를 넘는다.",
            "scope": "synthetic noiseless self-consistent data; one CG update rate; 121 illuminations and all active detector modes",
            "reference": "known synthetic object and real-subspace normal equations",
            "timing_boundary": "cached geometry; per-update core; pixel-to-mode preprocessing recorded separately",
            "metrics": {
                "full_update_median_s": full_median_s,
                "full_update_p95_s": full_recon["iteration_core_timing"]["p95_s"],
                "full_update_hz": 1.0 / full_median_s,
                "full_rhs_plus_100_updates_s": full_recon[
                    "reconstruction_core_s_including_rhs"
                ],
                "full_final_object_nrmse": full_recon["final"]["object_nrmse"],
                "full_final_data_residual": full_recon["final"]["data_residual"],
                "slab128_update_median_s": slab128_median_s,
                "slab128_update_hz": 1.0 / slab128_median_s,
            },
            "claim_boundary": odt_reconstruction["claim_boundary"],
            "reproduction": {
                "mode": "full saved synthetic reconstruction benchmark",
                "driver": "scripts/benchmark_odt_full_slab_reconstruction.py",
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/odt_full_slab_reconstruction_claim.json", "timing and inverse metrics"),
                    ("scripts/benchmark_odt_full_slab_reconstruction.py", "benchmark driver"),
                    ("tests/test_odt_full_slab_reconstruction.py", "regression tests"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "odt_cartesian_detector_remap_separate_branch",
            "application": "ODT",
            "status": "verified_separate_branch_integration_open",
            "headline": "Cartesian camera to banded angular sampling branch: z=1/z=8 remap-inclusive 12.52/11.11 Hz",
            "claim_ko": "Cartesian complex camera field를 cached bilinear remap으로 banded angular samples로 변환하는 별도 branch가 10 Hz를 넘고 ideal-banded reconstruction과 약 0.098% 이내로 일치한다.",
            "scope": "121 views; 320x320 Cartesian complex field; inner 96x128 plus outer 64x256 bands",
            "reference": "ideal banded angular input generated from the same synthetic object",
            "timing_boundary": "cached remap plus branch-specific reconstruction update; acquisition and hologram demodulation excluded",
            "metrics": {
                "samples_per_view": sum(band["samples_per_view"] for band in odt_detector["bands"]),
                "z1_total_s": z1_total_s,
                "z1_total_hz": 1.0 / z1_total_s,
                "z1_reconstruction_difference": detector_z1[
                    "remap_vs_ideal_reconstruction_rel_l2"
                ],
                "z8_total_s": z8_total_s,
                "z8_total_hz": 1.0 / z8_total_s,
                "z8_reconstruction_difference": detector_z8[
                    "remap_vs_ideal_reconstruction_rel_l2"
                ],
            },
            "claim_boundary": odt_detector["claim_boundary"]
            + [
                "This detector branch has not yet been measured with the final H28/rank16/adaptive-L packed operator.",
                "Do not combine its 12.52 Hz with the separate 81.24x packed/cuFINUFFT row.",
            ],
            "reproduction": {
                "mode": "full saved Cartesian-remap reconstruction driver",
                "driver": "scripts/benchmark_odt_banded_cartesian_reconstruction.py",
            },
            "artifacts": claim_artifacts(
                [
                    ("benchmark_results/odt_banded_cartesian_reconstruction.json", "accuracy and solve timing"),
                    (
                        "benchmark_results/odt_banded_cartesian_remap_hot_recheck.json",
                        "hot remap timing recheck",
                    ),
                    ("scripts/benchmark_odt_banded_cartesian_reconstruction.py", "benchmark driver"),
                    ("tests/test_odt_detector_geometry.py", "geometry regression tests"),
                ],
                release_entries,
            ),
        },
        {
            "claim_id": "general_curvature_frozen_holdout",
            "application": "general curvature / nonlinear optics",
            "status": "verified_single_frozen_holdout",
            "headline": "38 deg calibration to 25 deg no-refit holdout: 10/10 gates PASS",
            "claim_ko": "사전에 고정한 38-degree point calibration을 25-degree pump holdout에 재적합 없이 적용한 one-way bulk SHG cascade가 모든 gate를 통과한다.",
            "scope": "homogeneous LiNbO3; one-way pump to impressed SH current; one preselected r24 holdout",
            "reference": "direct tensor contraction and exact injected Yee-source modal contraction, plus SH impressed-current FDTD",
            "timing_boundary": "accuracy validation; pump and SH FDTD runtimes recorded but no performance claim",
            "metrics": {
                "cell_center_acfo_vs_direct_l2": curvature_row["cell_center_source"][
                    "acfo_vs_direct_relative_l2"
                ],
                "exact_yee_acfo_vs_direct_l2": curvature_row["exact_yee_source"][
                    "acfo_vs_direct_relative_l2"
                ],
                "source_transfer_direct_l2": curvature_row["exact_yee_source"][
                    "direct_vs_cell_center_direct_relative_l2"
                ],
                "sh_fdtd_vs_exact_yee_direct_l2": curvature_inner_shell[
                    "fdtd_vs_exact_yee_direct_relative_l2"
                ],
                "sh_fdtd_vs_exact_yee_acfo_l2": curvature_inner_shell[
                    "fdtd_vs_exact_yee_acfo_relative_l2"
                ],
                "ordinary_l2": curvature_inner_shell[
                    "branch_fdtd_vs_exact_yee_direct_relative_l2"
                ]["ordinary"],
                "extraordinary_l2": curvature_inner_shell[
                    "branch_fdtd_vs_exact_yee_direct_relative_l2"
                ]["extraordinary"],
                "monitor_shell_l2": curvature_row["calibrated_inner_outer_relative_l2"],
                "passed_gate_count": sum(curvature["gates"].values()),
                "total_gate_count": len(curvature["gates"]),
            },
            "claim_boundary": curvature["claim_boundary"],
            "reproduction": {
                "mode": "frozen point calibration plus one-way PyMeep cascade",
                "driver": "scripts/probe_linbo3_one_way_shg_cascade.py",
                "holdout_contract": curvature["holdout_contract"],
            },
            "artifacts": claim_artifacts(
                [
                    (
                        "benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24.json",
                        "frozen holdout metrics and gates",
                    ),
                    (
                        "benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24_fields_r24.npz",
                        "pump, nonlinear-current and SH field arrays",
                    ),
                    ("benchmark_results/linbo3_point_modal_calibration.json", "frozen training calibration"),
                    ("scripts/probe_linbo3_one_way_shg_cascade.py", "cascade driver"),
                    ("tests/test_general_curvature_publication_controls.py", "publication controls"),
                    (
                        "reports/general_curvature_frozen_holdout_ko/ACFO_general_curvature_frozen_holdout_validation_ko.pdf",
                        "verified Korean report",
                    ),
                ],
                release_entries,
            ),
        },
    ]

    open_gates = [
        {
            "gate_id": "odt_remap_final_packed_integration",
            "priority": 1,
            "status": "closed",
            "reason": "The Cartesian camera, cached remap, mode reduction and final packed operator now pass in one frozen run.",
            "result": {
                "worst_operator_rel_l2": max(
                    case["operator_errors_vs_h36"]["worst_rel_l2"]
                    for case in odt_integrated["cases"]
                ),
                "worst_remap_reconstruction_rel_l2": max(
                    case["remap_vs_ideal_reconstruction_rel_l2"]
                    for case in odt_integrated["cases"]
                ),
                "full_256z_steady_hot_hz": next(
                    case["integrated_steady_updates_per_second"]
                    for case in odt_integrated_timing["cases"]
                    if case["selected_n_z"] == 256
                ),
            },
            "acceptance": [
                "one run uses Cartesian camera-like complex fields, cached remap and final H28/rank16/adaptive-L forward/adjoint",
                "direct structured complex error remains <= 2e-6",
                "remap-vs-ideal reconstruction difference is reported",
                "preprocessing, pair/update timing and peak memory are reported from the same run",
            ],
        },
        {
            "gate_id": "odt_cufinufft_matched_effective_error",
            "priority": 2,
            "status": "closed_with_separate_process_caveat",
            "reason": "An independent complex128 exponent sum fixes the matched point at cuFINUFFT complex128 eps=1e-7; full timing is separate-process because both plans do not coexist in 8 GB.",
            "result": {
                "acfo_complex64_worst_direct_l2": odt_matched_c128["acfo"][
                    "worst_rel_l2_vs_direct"
                ],
                "matched_cufinufft_dtype": "complex128",
                "matched_cufinufft_eps": odt_matched_c128[
                    "matched_error_selection"
                ]["eps"],
                "matched_full_pair_median_s": odt_matched_full["cufinufft"][
                    "pair_median_s"
                ],
                "acfo_speedup_vs_matched_cufinufft": odt_matched_full[
                    "speedup_acfo_vs_matched_cufinufft"
                ],
            },
            "acceptance": [
                "small direct-reference forward and adjoint cases use identical q/object definitions",
                "ACFO threshold and cuFINUFFT tolerance are swept",
                "runtime is compared at matched measured complex error",
                "GPU residency, plan reuse, setpts and synchronization boundaries remain explicit",
            ],
        },
        {
            "gate_id": "odt_temporal_final_operator",
            "priority": 3,
            "status": "closed_frozen_noiseless_sequence",
            "reason": "An eight-frame physical Cartesian-camera sequence now runs through the integrated final packed path with warm/cold/reference rows.",
            "result": {
                "warm_1_update_median_hot_hz": next(
                    row["median_hot_hz"]
                    for row in odt_temporal["summary_rows"]
                    if row["mode"] == "warm_start" and row["updates"] == 1
                ),
                "warm_1_update_mean_object_rel_l2": next(
                    row["mean_object_rel_l2"]
                    for row in odt_temporal["summary_rows"]
                    if row["mode"] == "warm_start" and row["updates"] == 1
                ),
                "warm_3_update_vs_reference_ratio": next(
                    row["mean_object_error_vs_reference_ratio"]
                    for row in odt_temporal["summary_rows"]
                    if row["mode"] == "warm_start" and row["updates"] == 3
                ),
            },
            "acceptance": [
                "a short frozen sequence first verifies cold/warm equivalence and latency",
                "updates per frame and object/tracking error are recorded",
                "only after the short gate passes is a longer sequence justified",
            ],
        },
        {
            "gate_id": "independent_machine_replication",
            "priority": 4,
            "status": "external_pending",
            "reason": "All current production timings are from the same physical RTX 2070 SUPER machine.",
            "acceptance": [
                "rerun frozen commands on a distinct machine fingerprint",
                "preserve raw timing distributions, environment metadata and source hashes",
            ],
        },
        {
            "gate_id": "waxs_protein_exact_beta_followup",
            "priority": 5,
            "status": "closed_no_additional_rerun",
            "reason": "Existing 8008-atom protein direct validation and 216216/1001000-atom ordered-crystal controls already cover the chosen protein-crystal object.",
            "result": {
                "decision": waxs_protein_decision["decision"],
                "experimental_object": waxs_protein_decision[
                    "experimental_object_definition"
                ],
                "protein_unit_exact_beta_complex_l2": waxs_protein_decision[
                    "metrics"
                ]["protein_unit_cell"]["exact_beta_complex_l2_vs_direct"],
                "million_atom_direct_subset_complex_l2": waxs_protein_decision[
                    "metrics"
                ]["ordered_protein_supercells"][
                    "5x5x5_direct_subset_complex_l2"
                ],
            },
            "acceptance": [
                "predeclare the protein crystal/oriented nanocrystal object and detector contract",
                "compare an exact-atom subset and report performance without implying universal advantage",
            ],
        },
    ]

    gate_closure_artifacts = claim_artifacts(
        [
            (
                "benchmark_results/odt_banded_cartesian_final_packed_probe.json",
                "integrated accuracy gate",
            ),
            (
                "benchmark_results/odt_banded_cartesian_final_packed_full_timing.json",
                "integrated 64/128/256-z timing",
            ),
            (
                "benchmark_results/odt_cufinufft_matched_error_direct_subset.json",
                "same-dtype direct audit",
            ),
            (
                "benchmark_results/odt_cufinufft_matched_error_direct_subset_c128.json",
                "matched-error direct audit",
            ),
            (
                "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json",
                "matched-error production timing",
            ),
            (
                "benchmark_results/odt_cufinufft_c128_full_plan_diagnostic.json",
                "8 GB co-residency diagnosis",
            ),
            (
                "benchmark_results/odt_banded_cartesian_temporal_warm_start.json",
                "integrated temporal gate",
            ),
            (
                "benchmark_results/waxs_protein_exact_beta_followup_decision.json",
                "WAXS follow-up decision",
            ),
        ],
        release_entries,
    )
    all_artifacts = [item for claim in claims for item in claim["artifacts"]]
    all_artifacts.extend(gate_closure_artifacts)
    unique_artifacts = {item["path"]: item for item in all_artifacts}
    release_included = sum(
        bool(item["release_candidate"]["included"]) for item in unique_artifacts.values()
    )
    release_matching = sum(
        item["release_candidate"]["hash_matches_current"] is True
        for item in unique_artifacts.values()
    )

    return {
        "schema": "acfo-claim-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Map each paper-facing claim to its scope, timing boundary, reference, raw evidence, reproduction driver and unresolved gate without changing validated execution paths.",
        "worktree": git_snapshot(),
        "prior_release_candidate": release_metadata,
        "claim_count": len(claims),
        "claims": claims,
        "open_gates": open_gates,
        "gate_closure_artifacts": gate_closure_artifacts,
        "artifact_audit": {
            "unique_artifact_count": len(unique_artifacts),
            "included_in_prior_release_count": release_included,
            "included_and_hash_matching_count": release_matching,
            "not_in_prior_release_count": len(unique_artifacts) - release_included,
            "hash_changed_since_prior_release_count": release_included - release_matching,
        },
    }


def format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# ACFO claim-artifact 동결 manifest",
        "",
        "이 문서는 기존 검증 경로를 재구성하지 않고, 논문에 사용할 수 있는 주장과 실제 근거 파일을 연결한다.",
        "",
        "## 현재 snapshot",
        "",
        f"- git commit: `{manifest['worktree']['commit']}`",
        f"- tracked changes / untracked entries: `{manifest['worktree']['tracked_changes']} / {manifest['worktree']['untracked_entries']}`",
        f"- 핵심 claim 수: `{manifest['claim_count']}`",
        f"- 고유 근거 파일 수: `{manifest['artifact_audit']['unique_artifact_count']}`",
        f"- 이전 release에 포함 / 현재 hash 일치: `{manifest['artifact_audit']['included_in_prior_release_count']} / {manifest['artifact_audit']['included_and_hash_matching_count']}`",
        "",
        "## Claim 요약",
        "",
        "| ID | 영역 | 상태 | 핵심 결과 |",
        "|---|---|---|---|",
    ]
    for claim in manifest["claims"]:
        lines.append(
            f"| `{claim['claim_id']}` | {claim['application']} | `{claim['status']}` | {claim['headline']} |"
        )

    for claim in manifest["claims"]:
        lines.extend(
            [
                "",
                f"## {claim['claim_id']}",
                "",
                claim["claim_ko"],
                "",
                f"- 범위: {claim['scope']}",
                f"- reference: {claim['reference']}",
                f"- 측정 경계: {claim['timing_boundary']}",
                f"- 재현 driver: `{claim['reproduction'].get('driver', claim['reproduction'].get('accuracy_driver'))}`",
                "- 주요 수치:",
            ]
        )
        for key, value in claim["metrics"].items():
            rendered_value = format_metric(value)
            if rendered_value:
                lines.append(f"  - `{key}`: `{rendered_value}`")
        lines.append("- claim boundary:")
        for boundary in claim["claim_boundary"]:
            lines.append(f"  - {boundary}")
        lines.append("- 근거 파일:")
        for item in claim["artifacts"]:
            release = item["release_candidate"]
            if not release["included"]:
                frozen = "prior release 미포함"
            elif release["hash_matches_current"]:
                frozen = "prior release hash 일치"
            else:
                frozen = "prior release 이후 변경"
            lines.append(f"  - `{item['path']}` — {item['role']}; {frozen}")

    lines.extend(
        [
            "",
            "## 순차적으로 닫을 gate",
            "",
            "| 우선순위 | Gate | 상태 | 이유 |",
            "|---:|---|---|---|",
        ]
    )
    for gate in sorted(manifest["open_gates"], key=lambda item: item["priority"]):
        lines.append(
            f"| {gate['priority']} | `{gate['gate_id']}` | `{gate['status']}` | {gate['reason']} |"
        )
    lines.extend(
        [
            "",
            "첫 계산 gate는 `odt_remap_final_packed_integration`이다. 이 gate가 통과하기 전에는 detector-remap 처리율과 final packed/cuFINUFFT speedup을 하나의 pipeline 결과로 결합하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    manifest = build_manifest()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "claim_count": manifest["claim_count"],
                "artifact_audit": manifest["artifact_audit"],
                "open_gate_order": [gate["gate_id"] for gate in manifest["open_gates"]],
                "outputs": [
                    OUTPUT_JSON.relative_to(ROOT).as_posix(),
                    OUTPUT_MD.relative_to(ROOT).as_posix(),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
