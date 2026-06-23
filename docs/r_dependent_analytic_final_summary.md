# R-Dependent Analytic Final Summary

This note summarizes the current CPU default and the final direct scaling
measurement for the WAXS cake-map solver.

## Current Recommendation

Use R-dependent analytic cake as the conservative CPU default:

```python
plan.circular_fft_r_dependent_bandlimit(
    margin=16,
    cutoff_bin_size=16,
    analytic_kernel=True,
)
```

The most useful optimization is the half-spectrum no-copy path. If a full
circular FFT cache is already present, the C++ half-spectrum contraction now
reads the positive modes directly from that full FFT instead of materializing a
second compact positive-mode copy.

For strict working-set constraints, use:

```python
plan.circular_fft_r_dependent_bandlimit(
    margin=16,
    cutoff_bin_size=16,
    analytic_kernel=True,
    fused_analytic_kernel=True,
)
```

This generates the Miller analytic kernel inside the C++ contraction and avoids
storing compact `Khat(q,R,h)`. It is the better memory-constrained or GPU-design
candidate, while the no-copy path remains the fastest CPU default in the
prewarmed dense-baseline benchmark.

Do not make `r_block_size` the CPU default. It recomputes rFFT slabs and is
slower when a full or half FFT cache is already resident. It remains useful as a
working-set experiment for GPU or streaming implementations.

## 1M High-Q Memory Result

Conditions:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --skip-nufft --measure-memory --benchmark-r-dependent-cake --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --r-dependent-analytic-kernel
```

| R-dependent analytic variant | first total | cached total | method first | method cached | peak RSS delta |
|---|---:|---:|---:|---:|---:|
| no-copy positive modes | 0.472 s | 0.445 s | 0.350 s | 0.323 s | 40.5 MiB |
| R-block 16 | 1.024 s | 1.002 s | 0.921 s | 0.900 s | 52.8 MiB |
| fused Miller | 0.522 s | 0.455 s | 0.385 s | 0.319 s | 0.9 MiB |
| fused Miller + R-block 16 | 0.967 s | 0.903 s | 0.856 s | 0.792 s | 52.0 MiB |

The old R-dependent analytic method peak was `254.7 MiB`; the no-copy path
reduces that to `40.5 MiB`, and fused Miller reduces the measured method peak to
`0.9 MiB`.

## Direct Algorithm Scaling

The final scaling check uses fresh `PreparedCakePlan` instances per algorithm,
so dense circular, R-dependent analytic, fused R-dependent analytic, and NUFFT
do not reuse each other's caches.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_algorithm_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --nufft-q-block-size 2 --out benchmark_results\algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json
```

A second solver-only check without NUFFT was also run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_algorithm_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --skip-nufft --out benchmark_results\algorithm_scaling_highq_direct_fftfriendly_skipnufft.json
```

The table below averages the dense/R-dependent/fused timings from those two
direct runs. NUFFT comes from the NUFFT-included run. The physical grid now
rounds the angular FFT size upward to an FFT-friendly even length after applying
the bandlimit and arc constraints.

| atoms | n_phi | dense circular | R-dependent analytic | fused Miller | chunked NUFFT | dense/Rdep | NUFFT/Rdep | intensity error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 960 | 0.293 s | 0.092 s | 0.068 s | 6.673 s | 3.20x | 72.8x | 1.8e-7 |
| 250k | 1250 | 0.633 s | 0.183 s | 0.151 s | 10.756 s | 3.46x | 58.7x | 2.7e-6 |
| 500k | 1600 | 1.201 s | 0.305 s | 0.280 s | 18.881 s | 3.93x | 61.8x | 3.7e-7 |
| 1M | 2000 | 2.196 s | 0.675 s | 0.699 s | 34.358 s | 3.25x | 50.9x | 5.3e-7 |

Approximate log-log exponents versus atom count over 100k to 1M:

| method | measured exponent |
|---|---:|
| dense circular solve | 0.88 |
| R-dependent analytic solve | 0.85 |
| fused Miller solve | 1.00 |
| chunked NUFFT | 0.71 |

These exponents are empirical over a narrow threaded CPU range, so they should
not be overinterpreted. The stable conclusion is the relative separation:
R-dependent analytic is about `3-4x` faster than exact dense circular on this
fresh direct high-q sweep, and `51-73x` faster than chunked NUFFT while keeping
the intensity error below about `3e-6`.

Internal timing confirms the 500k anomaly:

| atoms | n_phi | hhat_half rFFT | analytic Khat | C++ contraction | cutoff work with z |
|---:|---:|---:|---:|---:|---:|
| 500k | 1556 | 0.681 s | 0.032 s | 0.139 s | 337M |
| 1M | 1952 | 0.420 s | 0.057 s | 0.325 s | 661M |

The contraction work nearly doubles from 500k to 1M, and contraction time grows
accordingly. The reason the total R-dependent time does not grow is that the
500k source rFFT is slower than the 1M source rFFT. Since `1556 = 4 * 389`,
PocketFFT falls onto an unfavorable transform length. Repeating only the 500k
case with FFT-friendlier angular sizes gave:

| 500k n_phi | R-dependent analytic solve |
|---:|---:|
| 1556 | 0.701 s |
| 1568 | 0.311 s |
| 1600 | 0.26-0.29 s |

The default physical grid now applies this FFT-friendly rounding. The high-q
sequence becomes `n_phi = 960, 1250, 1600, 2000` for 100k, 250k, 500k, and 1M
atoms, respectively.

## Fixed-dq q-Range Scaling

The atom-count scaling above keeps the q grid fixed at `Nq=40`. For WAXS
experiments, holding `dq` fixed while extending the q range is often the more
relevant high-q stress test. The following run fixes the structure at 1M atoms
and uses the same spacing as the earlier `qmax=6.3`, `Nq=40` benchmark:

```text
qmin = 0.05 A^-1
dq = (6.3 - 0.05) / 39 = 0.160256 A^-1
```

The measured qmax values are aligned to that spacing.

| qmax | Nq | n_phi | targets | dense circular | R-dependent analytic | fused Miller | chunked NUFFT | NUFFT/Rdep |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.13 | 14 | 720 | 10,080 | 0.372 s | 0.155 s | 0.159 s | 3.925 s | 25.3x |
| 4.06 | 26 | 1280 | 33,280 | 0.994 s | 0.342 s | 0.369 s | 11.273 s | 33.0x |
| 6.30 | 40 | 2000 | 80,000 | 2.204 s | 0.650 s | 0.593 s | 32.702 s | 50.3x |
| 8.06 | 51 | 2500 | 127,500 | 3.101 s | 0.954 s | 0.738 s | 56.402 s | 59.1x |

Approximate exponents versus qmax over this range:

| quantity | exponent |
|---|---:|
| Nq | 0.97 |
| n_phi | 0.94 |
| target count `Nq*n_phi` | 1.91 |
| dense circular solve | 1.62 |
| R-dependent analytic solve | 1.36 |
| fused Miller solve | 1.16 |
| chunked NUFFT | 2.02 |

This is the clearest high-q separation so far. With fixed `dq`, the detector
target count grows almost as `qmax^2`, and chunked NUFFT follows that growth
closely in wall time. The R-dependent analytic solver also gets more expensive,
because the harmonic support grows with `qR`, but its measured growth is slower
over this range. The speedup therefore increases from about `25x` at
`qmax ~= 2.13 A^-1` to about `59x` at `qmax ~= 8.06 A^-1`.

This is still best framed as a WAXS-specific factorization advantage, not a
universal asymptotic-order claim. The method exploits cylindrical binning,
source FFT reuse, and R-dependent harmonic support; the high-q fixed-`dq` sweep
shows that these factors become increasingly valuable as angular sampling and q
sampling grow together.

## Why The Scaling Differs

With fixed bin width and density, the physical grid grows roughly with atom
count:

- `n_r ~ N^(1/3)`
- `n_z ~ N^(1/3)`
- `n_phi ~ N^(1/3)` at fixed qmax and angular rule
- total histogram bins `B = n_r n_z n_phi ~ N`

Dense circular exact:

- histogram: `O(N_atoms)`
- source FFT over phi rows: `O(B log n_phi)`
- q/R/z/phi contraction: `O(N_q B)`
- practical scaling: close to `N log N`, measured near `N^1.1`

R-dependent analytic:

- source rFFT: `O(B log n_phi)` on a fresh plan
- contraction: `O(N_q n_z sum_R H_R)`, where `H_R ~= q_perp R + margin`
- for this physical scaling, `sum_R H_R ~ n_r^2`, so the contraction is still
  approximately `O(N_q N)` but with a smaller constant than full-phi dense
  contraction
- practical scaling: similar exponent to dense circular, but consistently lower
  wall time because high harmonics are skipped at small and intermediate R

Fused Miller:

- same mathematical work as R-dependent analytic contraction
- avoids storing `Khat(q,R,h)`
- can be slightly faster or slower depending on cache and Miller recurrence
  cost, but has the best working-set behavior

Chunked NUFFT:

- computes the reference amplitude directly from atoms to detector targets
- target count is `N_q n_phi`, and source count is `N_atoms`
- the implementation is memory-safe with q-blocking, but the direct atom-source
  workload remains much larger than the binned circular/R-dependent workloads
  in this high-q cake-map setting

## Source Files

- `src/waxs_cake/solvers.py`
- `src/waxs_cake/_cpp_solvers.cpp`
- `scripts/benchmark_physical_scaling.py`
- `scripts/benchmark_algorithm_scaling.py`
- `tests/test_solvers.py`
- `benchmark_results/algorithm_scaling_highq_direct_nufft_qb2.json`
- `benchmark_results/algorithm_scaling_highq_direct_skipnufft_check.json`
- `benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json`
- `benchmark_results/algorithm_scaling_highq_direct_fftfriendly_skipnufft.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q2p13.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q4p06.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q6p30.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q8p06.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_nocopy_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_fused_memory.json`
