# WAXS JSR Claim and Benchmark Summary

Date: 2026-06-13

## Purpose

이 문서는 현재까지 구현/최적화한 WAXS simulator의 benchmark 결과와
Journal of Synchrotron Radiation(JSR)을 목표로 할 때의 논문 claim 범위를
정리한다.

핵심 결론은 다음이다.

- 첫 논문은 넓은 general Fourier method가 아니라 WAXS-specific atomistic
  curve/cake-map simulator로 좁히는 것이 가장 방어 가능하다.
- 1D curve는 amorphous 또는 azimuthally symmetric WAXS pattern을 빠르게
  계산하는 경로로 주장할 수 있다. Debye formula 대체 가능성은 크지만,
  논문용으로는 Debye benchmark를 추가해야 한다.
- 2D cake map은 anisotropy, speckle, partially crystalline, defected,
  amorphous + crystalline mixed structures를 atomistic level에서 빠르게
  시뮬레이션하는 방향이 가장 강하다.
- 현재 저장된 benchmark 기준으로 high-q physical 2D cake map에서는 NUFFT보다
  매우 큰 속도 이득이 있다. low-q에서는 이득이 작지만 여전히 경쟁 가능하다.

## Recommended Paper Claim

권장하는 주 claim:

```text
A WAXS-specific cylindrical-histogram and Ewald-ring circular-convolution
method enables fast atomistic simulation of 1D WAXS curves and anisotropic
2D cake maps on physically scaled curved-Ewald grids, with controlled
binning/bandlimit error and favorable scaling against direct and NUFFT
baselines.
```

한국어로 풀면:

```text
WAXS cake-map geometry에 특화된 cylindrical histogram + Ewald-ring circular
convolution을 사용하면, 원자 구조로부터 1D WAXS curve와 2D cake map을
빠르게 계산할 수 있고, 특히 high-q 및 20 nm급 physical box에서 일반 NUFFT보다
유리한 scaling을 보인다.
```

논문에서 피해야 할 claim:

- NUFFT를 일반적으로 대체한다.
- 모든 scattering geometry에 적용되는 general Fourier algorithm이다.
- 완전한 crystal 계산에서 Dirichlet kernel/reciprocal-lattice method를 대체한다.
- WAXS에서 많은 orientation sweep이 항상 필요하므로 reuse가 주된 장점이다.

더 강한 범위:

- amorphous/isotropic system의 1D curve fast simulation
- anisotropic/speckled/defected/partially ordered system의 2D WAXS cake map
- q range나 form factor가 바뀌는 경우 histogram 이후 재계산이 쉽다는 practical
  advantage
- XFEL/synchrotron single-shot template generation에서 high-q curved-Ewald
  target grid를 빠르게 생성한다는 computational advantage

## Current Method Scope

현재 가장 안정적인 default story는 다음 경로이다.

```text
atoms
  -> cylindrical histogram H(element, R, z, phi)
  -> FFT over phi
  -> Ewald-ring circular harmonic contraction
  -> A(q, phi) or I(q)
```

2D cake map:

```text
A(q, phi), I(q, phi) = |A(q, phi)|^2
```

1D curve:

```text
I(q) = mean_phi |A(q, phi)|^2
```

중요 구현 상태:

- `PreparedCakePlan` 중심 구조
- exact circular FFT path가 기본
- Jacobi-Anger path는 현재 default에서 제외
- 2D cake는 R-dependent harmonic bandlimit + analytic Miller kernel + compact
  half-spectrum C++ contraction이 현재 best path
- 1D curve는 R-grouped path가 일반 default, high-q에서는 R-dependent path가
  유리할 수 있음
- chunked NUFFT baseline을 구현해 1M high-q에서 single-call NUFFT memory failure를
  회피하는 비교가 가능해짐

## Benchmark Conditions

대표 benchmark 공통 조건:

- synthetic water-density box
- water-equivalent atom density: about `100.1037 atoms / nm^3`
- 1M atoms corresponds to box side about `21.54 nm`
- `bin_width_nm = 0.1`
- `qmin = 0.05 A^-1`
- `nq = 40`
- `nphi_detector = 180` is only a lower bound
- actual `n_phi` is selected from physical q/r bandlimit
- `wavelength_nm = 0.1`
- histogram backend: C++ float32
- atan2 angle LUT: cubic, size 32
- circular backend: C++/pybind11
- R-dependent cake settings: `margin=16`, `cutoff_bin_size=16`
- reported 2D total time includes histogram + plan + first R-dependent cake solve

주의:

- 몇몇 current optimized 2D files는 NUFFT를 다시 실행하지 않았다. 같은 physical
  geometry/settings의 기존 FINUFFT baseline 파일에서 NUFFT 시간을 가져와
  speedup을 계산했다.
- 1M high-q에서는 single-call FINUFFT가 memory failure를 보였고, fair comparison을
  위해 chunked NUFFT 결과를 별도 사용한다.
- 현재 error는 주로 dense circular cake reference 대비 relative L2 intensity
  error이다. 실제 논문에서는 real structures와 q-dependent form factor validation이
  추가되어야 한다.

## 2D Cake Map Results

### High-q physical WAXS, `qmax = 6.3 A^-1`

이 영역이 현재 가장 강한 결과이다. speedup은 같은 geometry의 저장된 FINUFFT
baseline 대비 계산했다.

| atoms | box side | n_phi | current R-dependent cake total | cached-total | FINUFFT baseline | speedup vs FINUFFT | intensity error vs dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 10.00 nm | 924 | 0.096 s | 0.090 s | 4.214 s | 43.7x | 2.45e-7 |
| 250k | 13.57 nm | 1242 | 0.164 s | 0.144 s | 23.699 s | 144.4x | 2.79e-6 |
| 500k | 17.09 nm | 1556 | 0.304 s | 0.276 s | 214.364 s | 705.0x | 5.02e-7 |
| 1M | 21.54 nm | 1952 | 0.508 s | 0.468 s | 30.923 s, chunked q-block=2 | 60.8x | 5.16e-7 |

Interpretation:

- 100k-500k high-q에서는 저장된 single-call FINUFFT baseline이 성공했지만 runtime이
  빠르게 증가했다.
- 1M high-q에서는 single-call FINUFFT memory failure가 있었고, chunked NUFFT로
  회피하면 약 `30.9 s`가 걸렸다.
- current R-dependent cake path는 1M high-q에서 total `0.508 s`로, chunked
  NUFFT보다 약 `60.8x` 빠르다.
- 500k high-q에서 single-call FINUFFT baseline과 비교하면 약 `705x`로 보이지만,
  이 값은 NUFFT 내부 memory/grid 상황의 영향을 크게 받으므로 논문에서는 high-q
  scaling trend와 memory-aware/chunked baseline을 함께 제시해야 한다.

### Low-q physical WAXS, `qmax = 2.2 A^-1`

low-q에서는 NUFFT가 훨씬 덜 불리하다. 따라서 이 영역은 "항상 압도적으로 빠르다"는
주장이 아니라, "여전히 경쟁 가능하고 WAXS-specific output에서는 실용적이다"는
보조 근거로 사용하는 것이 좋다.

| atoms | n_phi | R-dependent cake total | FINUFFT baseline | speedup vs FINUFFT | intensity error vs dense |
|---:|---:|---:|---:|---:|---:|
| 100k | 344 | 0.095 s | 0.277 s | 2.92x | 2.17e-8 |
| 250k | 456 | 0.259 s | 0.417 s | 1.61x | 1.76e-7 |
| 500k | 564 | 0.340 s | 0.682 s | 2.00x | 4.37e-8 |
| 1M | 704 | 0.476 s, current optimized | 1.365 s | 2.87x | 8.41e-8 |

Interpretation:

- low-q에서는 NUFFT baseline이 충분히 빠르다.
- 그래도 1M, 20 nm급 box에서 current optimized path가 약 `2.9x` 빠르다.
- high-q WAXS로 갈수록 uniform q/theta grid가 더 조밀해지고 NUFFT target/working
  grid 부담이 커지면서 WAXS-specific method의 장점이 커진다.

## 1D Curve Results

1D curve는 full cake map을 만들지 않고 radial/ring-averaged intensity만 계산할 수
있다는 점이 중요하다. 이는 amorphous object나 azimuthally symmetric WAXS pattern에
대해 Debye-type curve calculation을 대체할 가능성이 있다.

현재 저장된 1D benchmark:

| atoms | qmax | n_phi | best 1D path total | NUFFT | speedup vs NUFFT | note |
|---:|---:|---:|---:|---:|---:|---|
| 100k | 2.2 A^-1 | 344 | 0.162 s | 0.282 s | 1.75x | R-grouped default wins |
| 100k | 6.3 A^-1 | 924 | 0.424 s | 4.362 s | 10.3x | R-dependent slightly faster |
| 1M | 2.2 A^-1 | 704 | 0.923 s | 1.631 s | 1.77x | R-grouped default wins |
| 1M | 6.3 A^-1 | 1952 | 2.295 s | single-call failed | n/a | R-dependent faster than default |

1D claim의 현재 상태:

- NUFFT 대비 1D WAXS-specific curve path는 이미 경쟁 가능하다.
- high-q에서는 R-dependent 1D path가 더 유리해진다.
- Debye formula와의 비교는 아직 논문용 benchmark로 정리되지 않았다.
- 따라서 논문에서는 "Debye replacement"를 최종 claim으로 쓰기 전에, small/medium
  N에서 direct Debye reference와 speed/error sweep을 추가해야 한다.

## Why The Method Becomes Faster

현재 해석은 다음이 가장 타당하다.

1. NUFFT는 general nonuniform Fourier transform이다.

   WAXS cake map의 target geometry가 Ewald ring/cake로 구조화되어 있어도, NUFFT는
   일반 target set으로 처리한다. high-q와 large box에서는 `Nq * Nphi` target 수와
   내부 working grid 부담이 커진다.

2. 이 방법은 WAXS geometry를 직접 이용한다.

   q point를 arbitrary target으로 보지 않고, 각 q에서 `q_perp` ring 위의 circular
   convolution으로 다룬다. 그래서 phi 방향은 FFT/harmonic contraction으로 처리된다.

3. R-dependent bandlimit이 불필요한 harmonic work를 줄인다.

   각 radial shell의 필요 harmonic cutoff는 대략 다음으로 제한된다.

   ```text
   h_max(q, R) = ceil(|q_perp(q)| * R + margin)
   ```

   outer radius가 large `n_phi`를 결정하더라도, 작은 R shell은 high harmonics를 모두
   계산할 필요가 없다.

4. 1D curve는 full cake output을 피할 수 있다.

   Parseval/ring-average 구조를 쓰면 full `A(q, phi)`를 만들지 않고도 `I(q)`를 얻을
   수 있다. 이 점은 amorphous/isotropic WAXS에서 특히 중요하다.

5. Histogram 이후 q/form-factor variation이 쉽다.

   구조 binning이 끝나면 q range, detector sampling, form factor weighting을 바꿔
   재계산하는 workflow에서 practical advantage가 있다. 다만 WAXS single-shot에서는
   반복 orientation sweep 자체를 핵심 장점으로 내세우면 약하다.

## Proposed JSR Paper Scope

추천 title 후보:

```text
Fast atomistic WAXS curve and cake-map simulation using cylindrical histograms
and Ewald-ring circular convolution
```

논문 contribution:

1. WAXS curved-Ewald cake-map grid에 특화된 cylindrical histogram formulation
2. 1D WAXS curve를 full cake map 없이 빠르게 계산하는 ring-averaged path
3. anisotropic 2D cake map을 위한 circular harmonic contraction
4. R-dependent harmonic bandlimit을 통한 high-q acceleration
5. direct/dense circular/NUFFT/chunked NUFFT 대비 speed-error benchmark
6. amorphous, defected, partially crystalline, mixed-order structures에 대한
   practical simulation scope

가장 좋은 narrative:

- SAXS처럼 orientation 변화가 큰 primary story가 아니라,
- WAXS single-shot/high-q/large atomistic box에서 detector-relevant cake/curve
  templates를 빠르게 생성하는 method이다.
- ideal crystal은 기존 reciprocal-lattice/Dirichlet kernel 방식이 강하므로, 우리의
  표적은 partially crystalline, defected, amorphous + crystalline, anisotropic
  disordered structures이다.

## Benchmark Suite Needed Before Manuscript

논문용으로 추가해야 할 benchmark는 다음 순서가 좋다.

1. Accuracy convergence

   - `bin_width_nm`
   - `margin`
   - `cutoff_bin_size`
   - `qmax`
   - atom count / physical box size
   - reference: direct small-N, dense circular, NUFFT

2. 1D curve benchmark against Debye

   - amorphous water/organic box
   - small-N direct Debye exact reference
   - medium-N chunked Debye or sampled-pair reference
   - compare speed and intensity error
   - clarify when azimuthal/ring average matches Debye orientation average

3. 2D cake benchmark against NUFFT

   - low-q and high-q
   - NUFFT single-call where memory permits
   - chunked NUFFT where memory does not permit
   - report memory and runtime separately

4. Physical examples

   - amorphous liquid/solution
   - anisotropic nanoparticle or fibril-like model
   - defected lattice
   - partially crystalline + amorphous composite

5. Form-factor variation

   - element-specific q-dependent form factors
   - changing q range after histogram construction
   - multi-element histogram timing

6. Reproducibility

   - save exact benchmark commands
   - save JSON outputs
   - include CPU/thread information
   - separate one-shot total, cached solve, histogram, and NUFFT memory behavior

## Bottom Line

현재 결과만 놓고 보면 JSR 방법론 논문 가능성은 충분하다. 다만 claim은 좁게 잡아야
한다.

가장 방어 가능한 claim:

```text
For physically scaled atomistic WAXS curve and cake-map simulation on
curved-Ewald grids, the cylindrical-histogram/circular-convolution method gives
controlled-error outputs and can substantially outperform generic NUFFT
baselines, especially for high-q 2D cake maps and large boxes.
```

현재 benchmark가 뒷받침하는 범위:

- 1D curve: NUFFT 대비 경쟁 가능, Debye 대체 claim은 추가 benchmark 필요
- 2D cake: high-q physical regime에서 매우 강함
- 1M, 20 nm급 box: chunked NUFFT와 비교해도 high-q 2D cake에서 약 `60x` faster
- low-q: 약 `2-3x` 수준으로 modest하지만 여전히 유리
- novelty: general algorithm novelty보다는 WAXS-specific computational method
  novelty가 강함

따라서 첫 논문은 "WAXS-specific fast simulator with controlled accuracy"로 가고,
MD convergence, incremental update, broader curved-Ewald structured sampling은 후속
논문 또는 discussion/future work로 남기는 것이 좋다.

## Source Benchmark Files

Main 2D files:

- `benchmark_results/physical_scaling_100k_bin0p1_q6p3A_rdep_cake_compact_tiled_m16_b16.json`
- `benchmark_results/physical_scaling_250k_bin0p1_q6p3A_rdep_cake_current_vs_nufft_m16_b16.json`
- `benchmark_results/physical_scaling_500k_bin0p1_q6p3A_rdep_cake_current_vs_nufft_m16_b16.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q6p3A_rdep_cake_chunked_nufft_qb2_m16_b16.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q2p2A_rdep_cake_current_vs_nufft_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_success_q2p2A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_success_100k_q6p3A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_success_250k_q6p3A_rdep_cake_m16_b16.json`
- `benchmark_results/physical_scaling_nufft_probe_500k_q6p3A_rdep_cake_m16_b16.json`

Main 1D files:

- `benchmark_results/physical_scaling_100k_bin0p1_q2p2A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_100k_bin0p1_q6p3A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q2p2A_rdep_curve_cpp_loopopt2_m16_b16_with_nufft.json`
- `benchmark_results/physical_scaling_1m_bin0p1_q6p3A_rdep_curve_cpp_loopopt2_m16_b16.json`

Related notes:

- `docs/1d_curve_optimization_notes.md`
- `docs/2d_cake_nufft_success_benchmarks.md`
