# Algorithm Development and Optimization Summary

Date: 2026-06-15

## Purpose

This note summarizes the algorithmic development so far for the atomistic WAXS
cake-map solver. It is written as manuscript-strategy material: what was
implemented, what was optimized, where the speedup comes from, and how to frame
the approximately 60x separation from a memory-safe NUFFT baseline.

The current claim should remain WAXS-specific:

```text
For physically scaled atomistic WAXS cake maps on curved-Ewald ring grids,
cylindrical histograms plus circular-harmonic factorization produce
controlled-error 2D detector maps much faster than generic NUFFT baselines in
the high-q, large-box regime.
```

This is not yet a general NUFFT replacement claim. The defensible novelty is
that the WAXS measurement geometry has exploitable circular-harmonic structure.

## Solver Evolution

The code evolved through the following stages.

1. Direct phase-sum reference

   The direct method evaluates

   ```text
   A(q, phi) = sum_j f_j(q) exp(i q(q, phi) dot r_j)
   ```

   and is used only for small correctness checks because it scales directly
   with atom count times detector targets.

2. Generic NUFFT baseline

   FINUFFT type-3 was added as the fair generic nonuniform Fourier baseline.
   For large high-q WAXS grids, single-call NUFFT can hit memory limits, so the
   fair baseline is now chunked over q blocks:

   ```python
   nufft_amplitude_chunked(..., q_block_size=2)
   ```

   Chunking makes the baseline complete, but it does not use the ring/cake
   structure of the WAXS target grid.

3. Cylindrical histogram factorization

   Atom positions are binned in cylindrical coordinates:

   ```text
   atoms -> H(element, R, z, beta)
   H -> FFT_beta Hhat(element, R, z, h)
   Hhat + circular kernel -> Ahat(q, h)
   IFFT_h -> A(q, phi), I(q, phi) = |A(q, phi)|^2
   ```

   This turns many atom-level phase evaluations into a binned source FFT plus
   q/R/z/h contraction.

4. Prepared plan and cache structure

   `PreparedCakePlan` became the central object. It stores the q geometry,
   histogram FFTs, circular kernels where useful, and provides dense circular,
   R-dependent, 1D curve, sparse-source, and NUFFT comparison paths.

5. Histogram and input-path optimization

   The histogram path was moved toward C++/float32-friendly execution:

   - uniform-bin index arithmetic instead of search-heavy binning;
   - `bincount` and compact indexed element paths;
   - cubic `atan2` LUT with small table size for fast beta assignment;
   - C++ histogram backend for the main physical benchmarks.

6. Dense circular FFT reference

   The dense circular path is the exact binned circular reference. It still
   computes all angular modes on the selected `n_phi` grid. This is important
   because R-dependent paths are compared against it for controlled cutoff
   error.

7. R-dependent harmonic cutoff

   The main high-q acceleration is the observation that not every R shell needs
   every angular harmonic. The cutoff is approximately

   ```text
   |h| <= q_perp(q) * R + margin
   ```

   The current production setting is:

   ```python
   margin = 16
   cutoff_bin_size = 16
   analytic_kernel = True
   ```

   This keeps the full detector output but skips high harmonics for small and
   intermediate R shells.

8. Analytic Miller/Bessel kernel

   The sampled circular kernel was replaced in the R-dependent path by compact
   analytic coefficients:

   ```text
   K_h(q, R) = N_phi i^h J_h(q_perp R)
   ```

   This removes the need to FFT a sampled kernel for every selected q/R pair
   and is the source of the large high-q one-shot speedup.

9. No-copy half-spectrum and fused Miller variants

   The current conservative CPU default is the no-copy positive-mode path. If a
   full circular FFT source is already resident, the C++ half-spectrum
   contraction reads positive modes directly instead of materializing another
   compact copy.

   The memory-constrained variant is:

   ```python
   plan.circular_fft_r_dependent_bandlimit(
       margin=16,
       cutoff_bin_size=16,
       analytic_kernel=True,
       fused_analytic_kernel=True,
   )
   ```

   This fused Miller path generates the analytic kernel inside the C++
   contraction and avoids storing compact `Khat(q, R, h)`. In the 1M high-q
   memory run, the old R-dependent analytic method had about `254.7 MiB` peak
   method memory; the no-copy path reduced this to `40.5 MiB`, and fused Miller
   reduced the measured method peak to about `0.9 MiB`. The fused path is
   therefore the best GPU/streaming design candidate even when the no-copy path
   is the fastest CPU default.

10. FFT-friendly physical angular grid

    The physical grid first computes the required angular resolution from the
    WAXS bandlimit and arc constraints, then rounds upward to an FFT-friendly
    even length using `scipy.fft.next_fast_len`.

    This matters. A previous 500k high-q case used `n_phi = 1556`, where
    `1556 = 4 * 389`, an unfavorable transform length for PocketFFT. Repeating
    that case with friendlier lengths changed the R-dependent analytic solve
    from about `0.701 s` at `1556` to about `0.26-0.31 s` near `1568-1600`.

    The current high-q physical sequence is:

    | atoms | required high-q grid after FFT-friendly rounding |
    |---:|---:|
    | 100k | `n_phi = 960` |
    | 250k | `n_phi = 1250` |
    | 500k | `n_phi = 1600` |
    | 1M | `n_phi = 2000` |

## Current High-Q Benchmark Picture

The most useful headline is the 1M, `qmax = 6.3 A^-1`, `Nq = 40`,
`bin_width = 0.1 nm` physical-grid case.

With the no-copy R-dependent analytic path:

| method | total time | note |
|---|---:|---|
| R-dependent analytic no-copy | `0.516 s` | includes histogram + first R-dependent cake solve |
| chunked NUFFT, q-block=2 | `30.682 s` | memory-safe FINUFFT baseline |
| speedup | `59.4x` | total-time comparison |
| intensity error vs dense circular | `5.16e-7` | controlled cutoff error |

This is the cleanest "about 60x" result. It should be presented as a
memory-safe NUFFT comparison, not as a single-call NUFFT failure story.

The fresh-plan scaling benchmark with FFT-friendly angular grids gives the same
qualitative result. In method-solve time, the fused R-dependent path is roughly
`50-90x` faster than chunked NUFFT across 100k to 1M atoms; including histogram
and first-call total time, the separation is smaller but still large. The
stable manuscript wording should be "about one to two orders of magnitude in
the high-q physical WAXS regime", with the 1M no-copy case giving the concrete
`59.4x` number.

## Why NUFFT Separates From The WAXS-Specific Solver

NUFFT sees the WAXS cake as a large nonuniform target set:

```text
N_targets = Nq * Nphi
```

At high q and large box size, `Nphi` must increase because angular modes up to
roughly `qmax * Rmax` must be represented. Chunking over q avoids memory
failure, but the atom-to-target workload remains large.

The WAXS-specific solver instead uses:

```text
source histogram -> beta FFT -> R-dependent harmonic contraction
```

The histogram makes source reuse explicit. The R-dependent cutoff then reduces
the effective harmonic work from "all R shells use all h modes" to "each R shell
uses only the h modes it can physically support".

That is why the method does not simply win by implementation tuning. The
optimization exposes a smaller effective workload for this measurement
geometry.

## Nq Versus dq

For manuscript framing, `Nq` should not be treated as the main physical knob.
`Nq` is just the number of radial samples after choosing a q range and q
spacing:

```text
dq = (qmax - qmin) / (Nq - 1)
```

At fixed `qmax`, increasing `Nq` mainly adds radial samples. That cost is
important, but it is mostly a radial loop multiplier.

For WAXS experiments, the stronger stress test is usually fixed `dq` while
extending the measured q range. Then both of these grow together:

```text
Nq increases because the q range is longer.
Nphi increases because qmax * Rmax requires more angular harmonics.
```

In the 1M fixed-`dq = 0.160 A^-1` q-range sweep:

| qmax | Nq | n_phi | targets `Nq*n_phi` | fused Miller total | chunked NUFFT |
|---:|---:|---:|---:|---:|---:|
| 2.13 | 14 | 720 | 10,080 | `0.234 s` | `3.925 s` |
| 4.06 | 26 | 1280 | 33,280 | `0.466 s` | `11.273 s` |
| 6.30 | 40 | 2000 | 80,000 | `0.711 s` | `32.702 s` |
| 8.06 | 51 | 2500 | 127,500 | `0.846 s` | `56.402 s` |

The target count grows approximately like `qmax^2` because `Nq` and `n_phi`
both increase. Chunked NUFFT follows that target growth closely. The
R-dependent/fused path also becomes more expensive because harmonic support
grows with `qR`, but it grows more slowly in this sweep.

Therefore the correct wording is:

```text
The relevant high-q stress is not Nq alone. At fixed dq, extending qmax
simultaneously increases the number of q shells and the angular bandlimit, so
the detector target count grows much faster than a radial Nq-only sweep would
suggest. This is where the WAXS-specific factorization separates most clearly
from NUFFT.
```

## Default Claim For A Paper Draft

Recommended wording:

```text
The optimized solver combines cylindrical atom histograms, FFT-friendly
physical angular grids, R-dependent harmonic pruning, and analytic/fused
circular kernels. On a 1M-atom high-q WAXS cake-map benchmark, the resulting
R-dependent analytic path computes a controlled-error 2D map in about 0.5 s,
where a memory-safe chunked NUFFT baseline takes about 31 s, giving an
approximately 60x total-time separation. Fixed-dq q-range sweeps show that the
advantage grows when qmax is extended, because Nq and the required angular
bandlimit increase together.
```

Avoid stronger wording for now:

- do not say this replaces NUFFT generally;
- do not say the speedup is a universal asymptotic order improvement;
- do not use single-call NUFFT memory failure as the primary result;
- do not make `Nq` alone the scaling story.

## Source Artifacts

- `src/waxs_cake/solvers.py`
- `src/waxs_cake/_cpp_solvers.cpp`
- `src/waxs_cake/physical_scaling.py`
- `scripts/benchmark_physical_scaling.py`
- `scripts/benchmark_algorithm_scaling.py`
- `docs/r_dependent_analytic_final_summary.md`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_nocopy_vs_nufft.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_nocopy_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_fused_memory.json`
- `benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q2p13.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q4p06.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q6p30.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q8p06.json`
