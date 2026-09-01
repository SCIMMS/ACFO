# ODT 유망 최적화 최종 요약

## 결론

256 x 256 x 256 object, 121 illumination, 7,929,856 detector q-sample 조건에서 최종 production candidate는 다음 조합이다.

- harmonic cutoff: `H=28` (57 H modes)
- axis illumination: algebraically exact `L=0` pruning (65 -> 1 L mode)
- effective axial operator: SVD `rank=16`
- ring transverse support: adaptive-L packed threshold `1e-6`
- ring active `(r,L)` fraction: `0.5046875` (`8,398 / 16,640` pairs)
- dtype: `complex64`
- GPU: NVIDIA GeForce RTX 2070 SUPER 8 GB

기존 exact/default 경로는 변경하지 않았으며, 모든 최적화는 명시적 opt-in flag일 때만 활성화된다.

## H=36 dense structured reference 직접 검증

최종 후보를 각 근사의 개별 오차를 더하는 방식이 아니라 H=36, full-rank, dense-L structured operator에 직접 비교했다.

| metric | value |
| --- | ---: |
| physical forward rel-L2 | `2.71677e-7` |
| physical adjoint rel-L2 | `2.21660e-7` |
| random-complex stress forward rel-L2 | `1.07211e-6` |
| random-complex stress adjoint rel-L2 | `9.53642e-7` |
| worst rel-L2 | `1.07211e-6` |
| forward/adjoint dot error | `6.98334e-7` |
| acceptance tolerance | `2e-6` |
| result | `PASS` |

동일 run의 hot forward+adjoint pair 중앙값은 reference `369.530 ms`, 최종 후보 `111.239 ms`로 `3.32194x` 빨랐다.

## Adaptive-L threshold sweep

H=28, rank=16 dense-L 후보를 기준으로 threshold별 packed support를 측정했다. Physical bead coefficient뿐 아니라 random-complex coefficient와 random-complex residual을 함께 사용했다.

| threshold | active fraction | pair median | dense 대비 | worst stress rel-L2 | pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| `1e-12` | `0.720673` | `138.407 ms` | `1.010x` | `5.256e-7` | yes |
| `1e-10` | `0.656490` | `129.791 ms` | `1.077x` | `5.478e-7` | yes |
| `1e-8` | `0.583293` | `117.943 ms` | `1.185x` | `5.418e-7` | yes |
| `1e-6` | `0.504688` | `104.294 ms` | `1.340x` | `9.616e-7` | yes |
| `1e-5` | `0.461899` | `97.018 ms` | `1.440x` | `8.630e-6` | no |
| `1e-4` | `0.416587` | `88.756 ms` | `1.575x` | `8.753e-5` | no |

따라서 `1e-6`이 설정한 accuracy tolerance를 통과하는 최속 threshold이고, `1e-5`부터는 stress input에서 명확히 실패한다.

## cuFINUFFT reusable-plan hot 비교

두 backend 모두 입력이 GPU에 있고, geometry와 plan을 반복 사용하며, cuFINUFFT의 makeplan/setpts와 ACFO setup은 hot time에서 제외했다. 각 backend를 사전 초기화하고 모든 plan-owned CUDA stream을 device-wide synchronize한 뒤 5 warm-up + 30 measured AB/BA 순서로 교차 측정했다.

| backend | pair median | IQR | min-max |
| --- | ---: | ---: | ---: |
| final ACFO candidate | `111.366 ms` | `111.147-111.630 ms` | `110.020-113.552 ms` |
| cuFINUFFT 2.5.1, eps=1e-6 | `9.04750 s` | `9.04054-9.06181 s` | `9.03247-9.13621 s` |

- robust pair speedup: `81.2411x`
- ACFO distribution: mean `111.393 ms`, standard deviation `0.716 ms`
- cuFINUFFT distribution: mean `9.05550 s`, standard deviation `24.012 ms`
- ACFO operator-plan setup: `0.600 s` (shared context construction 제외)
- cuFINUFFT plan + setpts actual synchronized setup: `16.290 s`
- interleaved process memory: PyTorch peak allocated `1,947.60 MiB`; CuPy pool `1,454.5 MiB`

cuFINUFFT와 ACFO 출력의 상호 rel-L2 차이는 forward `6.9906e-5`, adjoint `8.3587e-5`였다. 이것은 서로 다른 두 근사 backend의 교차 차이이며 exact-reference error로 해석하지 않는다. 최종 후보의 accuracy 판정은 위 H=36 dense structured reference 직접 검증에 근거한다.

## 측정 프로토콜에서 발견하고 수정한 문제

cuFINUFFT `setpts`는 plan-owned CUDA stream에 비동기 작업을 남길 수 있다. 초기 baseline은 Python 반환 시점을 setup 종료로 기록해 남은 setup 작업과 다음 PyTorch timing이 겹쳤다. 또한 Windows/CUDA 환경에서는 cuFINUFFT가 첫 executable CUDA path가 되면 packed PyTorch 경로가 크게 느려지는 초기화 순서 효과가 관찰되었다.

최종 프로토콜은 다음을 적용했다.

1. PyTorch packed forward/adjoint를 먼저 실행해 FFT, cuBLAS, index kernels를 초기화한다.
2. cuFINUFFT plan과 setpts를 구성한다.
3. current-stream sync가 아니라 device-wide sync로 setup 완료를 확인한다.
4. 양쪽 warm-up 후 AB/BA 순서를 교대로 측정한다.

진단 run에서 ACFO pair는 cuFINUFFT plan 생성 전 `110.770 ms`, plan 생성 후 `111.019 ms`, cuFINUFFT 한 pair 실행 후 `110.683 ms`로 유지되어, 정상 동기화 시 plan 공존 자체가 병목이 아님을 확인했다.

## 구현 및 회귀 검증

- default exact path: unchanged
- opt-in flags: `--prune-axis-l0`, `--axial-lowrank-rank`, `--ring-adaptive-l-packed-threshold`
- full-volume forward/adjoint와 selected-z forward/adjoint가 같은 packed mask와 같은 low-rank operator를 사용한다.
- exact L=0 pruning, full-rank SVD equivalence, dot-product adjointness, selected-z restriction, packed exact-zero behavior, approximate packed adjointness를 포함한 관련 test: `14 passed`

## 정식 근거 파일

- `odt_final_packed_candidate_validation.json`: H=36 direct end-to-end validation
- `odt_adaptive_l_packed_sweep.json`: adaptive-L threshold sweep
- `odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json`: synchronized, pre-initialized 30-repeat AB/BA comparison
- `odt_cufinufft_co_residency_diagnostic.json`: plan coexistence and synchronization diagnostic

파일명에 `final`이 없고 ACFO가 2-3 s로 측정된 중간 AB/BA 결과는 asynchronous setup 또는 initialization-order contamination을 포함하므로 최종 판정에서 제외한다.
