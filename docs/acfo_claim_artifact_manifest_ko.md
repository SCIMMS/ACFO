# ACFO claim-artifact 동결 manifest

이 문서는 기존 검증 경로를 재구성하지 않고, 논문에 사용할 수 있는 주장과 실제 근거 파일을 연결한다.

## 현재 snapshot

- git commit: `af68ee6096185ef2086cec31e0f9d8f4928bae42`
- tracked changes / untracked entries: `19 / 219`
- 핵심 claim 수: `7`
- 고유 근거 파일 수: `38`
- 이전 release에 포함 / 현재 hash 일치: `10 / 9`

## Claim 요약

| ID | 영역 | 상태 | 핵심 결과 |
|---|---|---|---|
| `waxs_detector_aware_local` | WAXS | `verified_local_external_rerun_pending` | EIGER2 X 4M envelope의 Nq=512 local 10/30 protocol에서 1.976x |
| `waxs_dense_exact_beta_highq` | WAXS | `accuracy_verified_dense_nq512_not_competitive` | 50,430-atom TIP3P 20 frames의 high-q exact-beta 정확도 PASS; dense Nq=512 성능 FAIL |
| `aidt_gpu_resident_core` | aIDT | `verified_processing_core` | 24x700x700 input to 700x700x35 GPU-resident core: 97.002 ms, 10.31 Hz |
| `odt_final_packed_operator` | ODT | `operator_verified_local_timing_baseline_accuracy_open` | 256^3 final packed pair 111.366 ms; H36 structured gate PASS; local cuFINUFFT ratio 81.24x |
| `odt_full_slab_update_throughput` | ODT | `verified_synthetic_update_throughput` | full 256^3 update 8.46 Hz; known-support z=128 update 10.64 Hz |
| `odt_cartesian_detector_remap_separate_branch` | ODT | `verified_separate_branch_integration_open` | Cartesian camera to banded angular sampling branch: z=1/z=8 remap-inclusive 12.52/11.11 Hz |
| `general_curvature_frozen_holdout` | general curvature / nonlinear optics | `verified_single_frozen_holdout` | 38 deg calibration to 25 deg no-refit holdout: 10/10 gates PASS |

## waxs_detector_aware_local

동일 binned source와 detector-aware curved targets에서 prepared ACFO가 FINUFFT보다 약 1.98배 빠르다.

- 범위: RTX 2070 SUPER local CPU timing; Nq=512; geometric detector envelope; fixed source representation
- reference: FINUFFT on the same binned source and active curved targets
- 측정 경계: 10 warm-ups and 30 AB/BA alternating cached solves; setup excluded
- 재현 driver: `scripts/summarize_waxs_detector_aware.py`
- 주요 수치:
  - `acfo_median_s`: `21.6019`
  - `finufft_median_s`: `42.6853`
  - `ratio_of_medians_speedup`: `1.97599`
  - `paired_p05_speedup`: `1.91758`
  - `complex_l2`: `7.0280e-07`
  - `memory_reduction_ratio`: `5.14469`
- claim boundary:
  - This is same-bin operator evidence, not end-to-end exact-atom high-q accuracy.
  - Module gaps and beamstop are not included in the geometric-envelope detector tier.
  - Independent-machine timing remains pending.
- 근거 파일:
  - `benchmark_results/waxs_detector_aware_decision.json` — decision; prior release hash 일치
  - `benchmark_results/protein_nanocrystal_detector_eiger4m_216k_nq512_w10_r30_alternating.json` — raw timing and accuracy; prior release hash 일치
  - `scripts/summarize_waxs_detector_aware.py` — decision builder; prior release hash 일치
  - `docs/acfo_ncs_waxs_detector_aware_ko.md` — human-readable interpretation; prior release hash 일치

## waxs_dense_exact_beta_highq

Bounded-memory exact-beta 경로는 dense disordered coordinates를 direct NDFT 정확도로 재현하지만 dense Nq=512 workload에서는 FINUFFT보다 느리다.

- 범위: 8 nm TIP3P trajectory; q=5.0-6.3 A^-1; local CPU wall time
- reference: direct NDFT for correctness; reusable FINUFFT for Nq=512 cross-check
- 측정 경계: exact-coordinate contraction and harmonic evaluation; file loading/form-factor construction excluded in the scaling row
- 재현 driver: `scripts/validate_tip3p_dense_highq_exact_beta.py`
- 주요 수치:
  - `frames`: `20`
  - `atoms_per_frame`: `50430`
  - `max_complex_l2_vs_direct`: `2.4562e-12`
  - `median_direct_over_exact_beta`: `17.9066`
  - `nq512_exact_beta_s`: `38.3731`
  - `nq512_finufft_hot_median_s`: `0.387114`
  - `nq512_cross_complex_l2`: `3.9558e-07`
  - `nq512_comparative_performance_pass`: `FAIL`
- claim boundary:
  - This closes dense-disorder small-case correctness for the selected two high-q rows, not Nq=512 production timing.
  - The exact-beta path is exact-coordinate and bounded-memory but still scales with atom count, q rows, and retained harmonics.
  - The TIP3P model uses neutral xray_f0 form factors without anomalous dispersion or experimental background calibration.
  - Direct NDFT is the correctness oracle on small cases; this Nq=512 row is an optimized-method cross-check.
  - FINUFFT eps=1e-6 is a practical timing baseline, not a converged numerical oracle.
  - Timings and sampled RSS are local to this machine, build, thread policy, and run.
  - The full polar grid is used rather than a detector rectangle so both methods produce identical targets.
- 근거 파일:
  - `benchmark_results/tip3p_dense_highq_exact_beta_20frames.json` — accuracy; prior release hash 일치
  - `benchmark_results/tip3p_exact_beta_finufft_512.json` — dense performance; prior release hash 일치
  - `scripts/validate_tip3p_dense_highq_exact_beta.py` — accuracy driver; prior release hash 일치
  - `scripts/benchmark_tip3p_exact_beta_finufft_512.py` — FINUFFT comparison driver; prior release hash 일치

## aidt_gpu_resident_core

고정 geometry와 GPU-resident input의 public aIDT reconstruction core가 RTX 2070 SUPER에서 10 Hz를 넘는다.

- 범위: public 24-illumination aIDT transfer reconstruction; complex64; prepared support-transfer cache
- reference: validated public-data contract and dense transfer baseline
- 측정 경계: GPU core only; setup, output statistics, CPU preprocessing, acquisition and scheduling excluded
- 재현 driver: `scripts/benchmark_aidt_transfer_torch_gpu.py`
- 주요 수치:
  - `gpu_run_median_s`: `0.0970022`
  - `gpu_run_hz`: `10.309`
  - `gpu_setup_s`: `0.91436`
  - `n_illumination`: `24`
  - `depth_count`: `35`
  - `support_active_fraction`: `0.264304`
  - `torch_peak_allocated_mib`: `3849.37`
- claim boundary:
  - This is a GPU-resident processing-core result, not raw-acquisition-to-reconstruction end-to-end 10 Hz.
  - Output statistics were deliberately excluded in the frozen 10 Hz row.
- 근거 파일:
  - `benchmark_results/aidt_10hz_full700_opt_repeat.json` — timing and configuration; prior release 미포함
  - `validation/odt_aidt/aidt_diatom_public_contract_validation.json` — public-data contract validation receipt; prior release 미포함
  - `scripts/benchmark_aidt_transfer_torch_gpu.py` — benchmark driver; prior release hash 일치
  - `benchmark_results/aidt_10hz_full700_opt_repeat.md` — human-readable summary; prior release 미포함

## odt_final_packed_operator

Final H28/rank16/adaptive-L packed operator는 H36 structured reference gate를 통과하고 local reusable-plan timing에서 큰 실행시간 차이를 보인다.

- 범위: 256^3; 121 illuminations; 7,929,856 q samples; fixed geometry; complex64
- reference: H36 full-rank dense-L structured operator for ACFO accuracy; cuFINUFFT reusable plan for timing
- 측정 경계: both inputs GPU-resident; setup/allocation/setpts excluded; 5 warm-ups plus 30 AB/BA pairs
- 재현 driver: `scripts/validate_odt_final_packed_candidate.py`
- 주요 수치:
  - `worst_rel_l2_vs_h36`: `1.0721e-06`
  - `acceptance_tolerance`: `2.0000e-06`
  - `dot_error`: `6.9833e-07`
  - `acfo_pair_median_s`: `0.111366`
  - `cufinufft_pair_median_s`: `9.0475`
  - `local_pair_speedup`: `81.2411`
  - `cross_backend_forward_rel_l2`: `6.9906e-05`
  - `cross_backend_adjoint_rel_l2`: `8.3587e-05`
- claim boundary:
  - The 81.24x value is a measured local hot-time ratio, not yet a matched-effective-error speedup.
  - The ACFO accuracy gate is against H36 structured ACFO; the cross-backend difference is not an independent exact error.
  - Independent-machine replication remains pending.
- 근거 파일:
  - `benchmark_results/odt_final_packed_candidate_validation.json` — structured accuracy; prior release 미포함
  - `benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json` — local hot timing; prior release 미포함
  - `scripts/validate_odt_final_packed_candidate.py` — accuracy driver; prior release 미포함
  - `scripts/benchmark_odt_cufinufft_gpu_baseline.py` — timing driver; prior release 이후 변경
  - `benchmark_results/odt_promising_optimization_final_summary_ko.md` — decision summary; prior release 미포함

## odt_full_slab_update_throughput

Final packed operator를 이용한 fixed-geometry normal-equation update가 full volume에서 8.46 Hz, z=128 known-support slab에서 10 Hz를 넘는다.

- 범위: synthetic noiseless self-consistent data; one CG update rate; 121 illuminations and all active detector modes
- reference: known synthetic object and real-subspace normal equations
- 측정 경계: cached geometry; per-update core; pixel-to-mode preprocessing recorded separately
- 재현 driver: `scripts/benchmark_odt_full_slab_reconstruction.py`
- 주요 수치:
  - `full_update_median_s`: `0.11819`
  - `full_update_p95_s`: `0.119013`
  - `full_update_hz`: `8.46095`
  - `full_rhs_plus_100_updates_s`: `11.904`
  - `full_final_object_nrmse`: `0.185692`
  - `full_final_data_residual`: `0.00176906`
  - `slab128_update_median_s`: `0.0940006`
  - `slab128_update_hz`: `10.6382`
- claim boundary:
  - The full case treats all 256 z planes as unknown.
  - Each slab case assumes that only the stated centered z support is unknown; outside-slab contributions are absent or already removed.
  - All illumination views and all active detector harmonics are used.
  - Pixel-to-mode conversion is a one-time exact normal-operator-preserving preprocessing step for repeated geometry.
  - The inversion is synthetic, noiseless, and self-consistent; independent operator accuracy is reported separately.
  - Object NRMSE and data residual are both reported because missing-cone conditioning limits what data consistency alone proves.
- 근거 파일:
  - `benchmark_results/odt_full_slab_reconstruction_claim.json` — timing and inverse metrics; prior release 미포함
  - `scripts/benchmark_odt_full_slab_reconstruction.py` — benchmark driver; prior release 미포함
  - `tests/test_odt_full_slab_reconstruction.py` — regression tests; prior release 미포함

## odt_cartesian_detector_remap_separate_branch

Cartesian complex camera field를 cached bilinear remap으로 banded angular samples로 변환하는 별도 branch가 10 Hz를 넘고 ideal-banded reconstruction과 약 0.098% 이내로 일치한다.

- 범위: 121 views; 320x320 Cartesian complex field; inner 96x128 plus outer 64x256 bands
- reference: ideal banded angular input generated from the same synthetic object
- 측정 경계: cached remap plus branch-specific reconstruction update; acquisition and hologram demodulation excluded
- 재현 driver: `scripts/benchmark_odt_banded_cartesian_reconstruction.py`
- 주요 수치:
  - `samples_per_view`: `28672`
  - `z1_total_s`: `0.0798874`
  - `z1_total_hz`: `12.5176`
  - `z1_reconstruction_difference`: `9.7245e-04`
  - `z8_total_s`: `0.0900205`
  - `z8_total_hz`: `11.1086`
  - `z8_reconstruction_difference`: `9.7657e-04`
- claim boundary:
  - The physical input remains a Cartesian complex camera field; banded polar samples are cached bilinear interpolants.
  - The direct Cartesian reference is generated with chunked cuFINUFFT and is validation-only setup work.
  - No acquisition transfer, hologram demodulation, or measurement noise is included.
  - This detector branch has not yet been measured with the final H28/rank16/adaptive-L packed operator.
  - Do not combine its 12.52 Hz with the separate 81.24x packed/cuFINUFFT row.
- 근거 파일:
  - `benchmark_results/odt_banded_cartesian_reconstruction.json` — accuracy and solve timing; prior release 미포함
  - `benchmark_results/odt_banded_cartesian_remap_hot_recheck.json` — hot remap timing recheck; prior release 미포함
  - `scripts/benchmark_odt_banded_cartesian_reconstruction.py` — benchmark driver; prior release 미포함
  - `tests/test_odt_detector_geometry.py` — geometry regression tests; prior release 미포함

## general_curvature_frozen_holdout

사전에 고정한 38-degree point calibration을 25-degree pump holdout에 재적합 없이 적용한 one-way bulk SHG cascade가 모든 gate를 통과한다.

- 범위: homogeneous LiNbO3; one-way pump to impressed SH current; one preselected r24 holdout
- reference: direct tensor contraction and exact injected Yee-source modal contraction, plus SH impressed-current FDTD
- 측정 경계: accuracy validation; pump and SH FDTD runtimes recorded but no performance claim
- 재현 driver: `scripts/probe_linbo3_one_way_shg_cascade.py`
- 주요 수치:
  - `cell_center_acfo_vs_direct_l2`: `0.00620652`
  - `exact_yee_acfo_vs_direct_l2`: `0.00798115`
  - `source_transfer_direct_l2`: `0.0270793`
  - `sh_fdtd_vs_exact_yee_direct_l2`: `0.0424627`
  - `sh_fdtd_vs_exact_yee_acfo_l2`: `0.0444861`
  - `ordinary_l2`: `0.0436366`
  - `extraordinary_l2`: `0.04225`
  - `monitor_shell_l2`: `0.0112784`
  - `passed_gate_count`: `10`
  - `total_gate_count`: `10`
- claim boundary:
  - p
  - u
  - m
  - p
  -
  - p
  - r
  - o
  - p
  - a
  - g
  - a
  - t
  - i
  - o
  - n
  -
  - i
  - s
  -
  - s
  - o
  - l
  - v
  - e
  - d
  -
  - b
  - y
  -
  - M
  - a
  - x
  - w
  - e
  - l
  - l
  -
  - F
  - D
  - T
  - D
  -
  - a
  - n
  - d
  -
  - f
  - e
  - e
  - d
  - s
  -
  - t
  - h
  - e
  -
  - S
  - H
  -
  - s
  - o
  - u
  - r
  - c
  - e
  - ,
  -
  - b
  - u
  - t
  -
  - t
  - h
  - e
  -
  - c
  - a
  - s
  - c
  - a
  - d
  - e
  -
  - r
  - e
  - m
  - a
  - i
  - n
  - s
  -
  - u
  - n
  - d
  - e
  - p
  - l
  - e
  - t
  - e
  - d
  -
  - a
  - n
  - d
  -
  - o
  - n
  - e
  - -
  - w
  - a
  - y
  - ;
  -
  - i
  - t
  -
  - e
  - x
  - c
  - l
  - u
  - d
  - e
  - s
  -
  - p
  - u
  - m
  - p
  -
  - b
  - a
  - c
  - k
  - -
  - a
  - c
  - t
  - i
  - o
  - n
  - ,
  -
  - c
  - o
  - u
  - p
  - l
  - e
  - d
  -
  - n
  - o
  - n
  - l
  - i
  - n
  - e
  - a
  - r
  -
  - t
  - i
  - m
  - e
  -
  - s
  - t
  - e
  - p
  - p
  - i
  - n
  - g
  - ,
  -
  - i
  - n
  - t
  - e
  - r
  - f
  - a
  - c
  - e
  - s
  - ,
  -
  - p
  - e
  - r
  - i
  - o
  - d
  - i
  - c
  -
  - p
  - o
  - l
  - i
  - n
  - g
  - ,
  -
  - l
  - o
  - s
  - s
  - ,
  -
  - a
  - n
  - d
  -
  - a
  - b
  - s
  - o
  - l
  - u
  - t
  - e
  -
  - c
  - o
  - n
  - v
  - e
  - r
  - s
  - i
  - o
  - n
  -
  - e
  - f
  - f
  - i
  - c
  - i
  - e
  - n
  - c
  - y
- 근거 파일:
  - `benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24.json` — frozen holdout metrics and gates; prior release 미포함
  - `benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24_fields_r24.npz` — pump, nonlinear-current and SH field arrays; prior release 미포함
  - `benchmark_results/linbo3_point_modal_calibration.json` — frozen training calibration; prior release 미포함
  - `scripts/probe_linbo3_one_way_shg_cascade.py` — cascade driver; prior release 미포함
  - `tests/test_general_curvature_publication_controls.py` — publication controls; prior release 미포함
  - `reports/general_curvature_frozen_holdout_ko/ACFO_general_curvature_frozen_holdout_validation_ko.pdf` — verified Korean report; prior release 미포함

## 순차적으로 닫을 gate

| 우선순위 | Gate | 상태 | 이유 |
|---:|---|---|---|
| 1 | `odt_remap_final_packed_integration` | `closed` | The Cartesian camera, cached remap, mode reduction and final packed operator now pass in one frozen run. |
| 2 | `odt_cufinufft_matched_effective_error` | `closed_with_separate_process_caveat` | An independent complex128 exponent sum fixes the matched point at cuFINUFFT complex128 eps=1e-7; full timing is separate-process because both plans do not coexist in 8 GB. |
| 3 | `odt_temporal_final_operator` | `closed_frozen_noiseless_sequence` | An eight-frame physical Cartesian-camera sequence now runs through the integrated final packed path with warm/cold/reference rows. |
| 4 | `independent_machine_replication` | `external_pending` | All current production timings are from the same physical RTX 2070 SUPER machine. |
| 5 | `waxs_protein_exact_beta_followup` | `closed_no_additional_rerun` | Existing 8008-atom protein direct validation and 216216/1001000-atom ordered-crystal controls already cover the chosen protein-crystal object. |

첫 계산 gate는 `odt_remap_final_packed_integration`이다. 이 gate가 통과하기 전에는 detector-remap 처리율과 final packed/cuFINUFFT speedup을 하나의 pipeline 결과로 결합하지 않는다.
