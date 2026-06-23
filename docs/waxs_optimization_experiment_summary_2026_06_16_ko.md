# WAXS Cake-Map Solver 최적화 실험 요약

작성일: 2026-06-16

## Executive Summary

- 직접 phase-sum과 FINUFFT type-3 baseline에서 시작해, cylindrical histogram, Ewald-ring circular FFT, R-dependent harmonic cutoff 구조로 solver를 재구성했다.
- 현재 가장 깔끔한 total-time headline은 1M atom, qmax=6.3 A^-1 case이다. R-dependent analytic no-copy는 0.516 s, chunked NUFFT는 30.682 s로 약 59.4x 빠르다.
- 최신 fused Miller solver-only 비교에서는 1M high-q에서 fused first/cached solve가 0.329 s / 0.306 s이고, chunked NUFFT가 32.499 s이다. representation build를 제외하면 약 98.9x / 106.4x 빠르다.
- 같은 최신 run에서 solve-call peak RSS delta는 fused Miller 0.5 MiB, chunked NUFFT 2142.7 MiB이다. 이 결과가 공간복잡도와 transient working-set 측면의 가장 강한 증거다.
- 현재 주장은 일반 NUFFT replacement가 아니라, WAXS cake-map의 curved-Ewald/circular structure를 이용한 domain-specific factorization으로 잡는 것이 가장 방어적이다.

## 최적화 진행 흐름

| 단계 | 핵심 변경 | 의미 |
|---:|---|---|
| 1 | Direct phase-sum reference | 소형 case 정확도 기준. O(N_atoms * targets). |
| 2 | FINUFFT type-3 + q-block chunking | generic baseline. 큰 high-q grid에서 single-call memory failure를 피하는 공정 비교 기준. |
| 3 | Cylindrical histogram H(e,R,z,beta) | 원자별 target 합산을 binned source FFT + q/R/z/h contraction으로 바꿈. |
| 4 | PreparedCakePlan cache | q geometry, z phase, histogram FFT, kernel path를 재사용 가능하게 분리. |
| 5 | C++/float32 histogram + angle LUT | representation build의 주된 비용을 줄이고 1M physical grid에서 0.1-0.2 s 수준으로 안정화. |
| 6 | R-dependent harmonic cutoff | abs(h) <= q_perp R + margin. 작은 R shell에서 불필요한 고조파 제거. |
| 7 | Analytic Miller/Bessel kernel | sampled kernel FFT 대신 K_h = N_phi i^h J_h(q_perp R) 사용. |
| 8 | No-copy positive half-spectrum | 추가 compact positive-mode copy 제거. 1M high-q method peak 254.7 -> 40.5 MiB. |
| 9 | Fused Miller kernel | Khat(q,R,h)를 저장하지 않고 contraction 내부에서 생성. solve working-set을 거의 제거. |
| 10 | FFT-friendly n_phi rounding | pathological n_phi=1556 회피. 500k solve 0.701 s -> 0.26-0.31 s. |

## High-q Direct Solver Comparison

동일한 high-q physical grid(qmax=6.3 A^-1, Nq=40)에서 fresh PreparedCakePlan을 사용해 dense circular, R-dependent analytic, fused Miller, chunked NUFFT를 비교했다.

| atoms | n_phi | dense circular | R-dependent | fused Miller | chunked NUFFT | NUFFT/R-dependent | I error vs dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 960 | 0.293 s | 0.092 s | 0.068 s | 6.673 s | 72.8x | 1.8e-7 |
| 250k | 1250 | 0.633 s | 0.183 s | 0.151 s | 10.756 s | 58.7x | 2.7e-6 |
| 500k | 1600 | 1.201 s | 0.305 s | 0.280 s | 18.881 s | 61.8x | 3.7e-7 |
| 1M | 2000 | 2.184 s | 0.638 s | 0.645 s | 34.358 s | 53.8x | 5.3e-7 |

해석: R-dependent analytic은 dense circular 대비 약 3-4x, chunked NUFFT 대비 약 51-73x 빠르다. intensity error는 dense circular 기준 대략 1e-7에서 3e-6 범위로 유지된다.

## 현재 Best Headline: 1M High-q

| metric | no-copy R-dependent | fused Miller | chunked NUFFT | 해석 |
|---|---:|---:|---:|---|
| total time | 0.516 s | 0.505 s | 32.499 s | one-shot fresh structure 비교 |
| solver-only first | 0.350 s | 0.329 s | 32.499 s | representation build 제외 |
| solver-only cached | 0.323 s | 0.306 s | 32.499 s | q/form-factor reuse 문맥 |
| solve peak delta | 40.5 MiB | 0.5 MiB | 2142.7 MiB | timed solve-call working set |
| I error vs dense | 5.16e-7 | 4.85e-7 | 5.55e-4 | controlled cutoff/reference gap |

요약: total-time으로는 약 60-64x, representation을 제외한 fused solver-only로는 약 99-106x 이득이다. 공간 측면에서는 fused Miller가 compact Khat tensor를 materialize하지 않아 solve-call 추가 peak가 약 0.5 MiB로 측정된다.

## Fixed-dq q-range Scaling

1M atom 구조를 고정하고 dq를 유지한 채 qmax를 늘리면 Nq와 n_phi가 함께 증가한다. 이 조건이 WAXS high-q stress를 가장 잘 보여준다.

| qmax (A^-1) | Nq | n_phi | targets | R-dependent | fused Miller | chunked NUFFT | NUFFT/R-dependent | NUFFT/fused |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.13 | 14 | 720 | 10,080 | 0.155 s | 0.159 s | 3.925 s | 25.3x | 24.7x |
| 4.06 | 26 | 1280 | 33,280 | 0.342 s | 0.369 s | 11.273 s | 33.0x | 30.5x |
| 6.30 | 40 | 2000 | 80,000 | 0.650 s | 0.593 s | 32.702 s | 50.3x | 55.1x |
| 8.06 | 51 | 2500 | 127,500 | 0.954 s | 0.738 s | 56.402 s | 59.1x | 76.4x |

해석: qmax가 2.13 -> 8.06 A^-1로 증가할 때 target count는 약 qmax^2에 가깝게 증가한다. NUFFT wall time도 이 target growth를 강하게 따라가지만, R-dependent/fused 경로는 harmonic support를 shell별로 잘라 더 완만하게 증가한다.

## Sparse-source / Combined Path 실험

별도로 sparse source projection과 R-dependent cutoff를 결합한 경로도 시험했다. 이 경로는 active (element,R,z,beta) bin을 (element,R) profile로 묶은 뒤 q-dependent source projection을 수행한다. 현재는 CPU default라기보다는 large-grid 및 GPU/streaming 후보로 보는 것이 맞다.

| case | dense | R-dependent | sparse source | combined | combined/R-dependent |
|---|---:|---:|---:|---:|---:|
| high-q 100k | 0.298 s | 0.193 s | 0.229 s | 0.156 s | 1.23x |
| high-q 1M | 2.479 s | 1.149 s | 1.296 s | 0.786 s | 1.46x |
| low-q 100k | 0.169 s | 0.075 s | 0.126 s | 0.061 s | 1.23x |
| low-q 1M | 0.769 s | 0.643 s | 0.516 s | 0.383 s | 1.68x |

combined path는 500k/1M 또는 low-q large-grid에서 유리한 구간이 보였지만, analytic fused R-dependent path의 memory 개선 이후에는 즉시 default로 바꾸기보다 후속 GPU/selected-h 설계 후보로 두는 편이 안전하다.

## 논문 관점의 현재 결론

- 시간복잡도 관점: generic NUFFT는 atom-source와 detector target 수에 직접적으로 민감하다. WAXS-specific solver는 binned source, beta FFT, shell별 harmonic cutoff로 effective workload를 줄인다.
- 공간복잡도 관점: fused Miller는 Khat(q,R,h)를 저장하지 않으므로 transient solver working set이 극적으로 줄어든다. 최신 1M high-q run에서 NUFFT 대비 peak delta가 2142.7 MiB vs 0.5 MiB로 갈린다.
- 정확도 관점: R-dependent/fused path의 intensity error는 dense circular 기준 대략 1e-7에서 1e-6 수준이다. NUFFT와의 차이는 NUFFT eps와 binned representation 차이가 함께 포함되므로 보조 지표로 쓰는 것이 좋다.
- claim boundary: Communications Physics급 1차 논문은 WAXS cake-map 특화 알고리즘으로 좁게 잡는 것이 가장 강하다. Nature Communications 이상으로 확장하려면 cross-field curved/Fourier-manifold validation이 필요하다.

## Source Artifacts

- docs/algorithm_development_optimization_summary.md
- docs/r_dependent_analytic_final_summary.md
- docs/sparse_source_r_dependent_combined_benchmark.md
- docs/r_dependent_fused_nufft_memory_solver_only_benchmark.md
- benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json
- benchmark_results/physical_scaling_highq_rdep_fused_nufft_memory_solver_only.json
- benchmark_results/qmax_scaling_1m_dq0p160_q*.json
