# Sparse Source + R-Dependent Combined Benchmark

Date: 2026-06-15

This benchmark tests the combined path:

```text
active (element, R, z, beta)
  -> q-dependent sparse source projection
  -> beta FFT
  -> C++ R-dependent harmonic cutoff contraction
  -> IFFT cake map
```

The API is:

```python
PreparedCakePlan.circular_fft_sparse_source_r_dependent(
    margin=16,
    cutoff_bin_size=16,
    profile_chunk_size=64,
)
```

It matches `PreparedCakePlan.circular_fft_r_dependent_bandlimit()` with the same
sampled kernel and cutoff settings up to complex64 accumulation differences.

The memory-optimized implementation avoids two large temporaries from the first
combined prototype:

- kernel FFTs are built only for active `R` values in the current sparse source
  chunk, instead of for the full `R` grid;
- the R-dependent cutoff is applied inside a C++ contraction loop, so the code
  no longer materializes a `(q_block, profile_chunk, h)` boolean mask.
- with `analytic_kernel=True`, selected `R` bins use compact analytic
  `K_h = Nphi i^h J_h(q_perp R)` coefficients instead of sampled kernel FFTs.

## Commands

High-q:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 2 --skip-nufft --benchmark-r-dependent-cake --benchmark-sparse-source-projection --benchmark-sparse-source-r-dependent --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_highq_sparse_source_rdep_combined_repeats2.json
```

Low-q:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 2.2 --q-unit inv_angstrom --nq 40 --repeats 2 --skip-nufft --benchmark-r-dependent-cake --benchmark-sparse-source-projection --benchmark-sparse-source-r-dependent --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_lowq_sparse_source_rdep_memory_optimized_repeats2.json
```

Memory runs use `--measure-memory`. To reduce allocator/order bias, each method
was run in a separate process, for example:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --skip-nufft --measure-memory --benchmark-sparse-source-r-dependent --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_highq_sparse_source_rdep_memory_isolated.json
```

Analytic-kernel high-q run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 2 --skip-nufft --benchmark-r-dependent-cake --benchmark-sparse-source-projection --benchmark-sparse-source-r-dependent --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --r-dependent-analytic-kernel --out benchmark_results\physical_scaling_highq_sparse_source_rdep_analytic_repeats2.json
```

## High-q, qmax = 6.3 A^-1

| atoms | n_phi | dense total | R-dependent total | sparse source total | combined total | combined vs dense | combined vs R-dependent | combined vs sparse source |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 924 | 0.298 s | 0.193 s | 0.229 s | 0.156 s | 1.91x | 1.23x | 1.46x |
| 250k | 1242 | 0.684 s | 0.291 s | 0.391 s | 0.349 s | 1.96x | 0.83x | 1.12x |
| 500k | 1556 | 1.817 s | 0.636 s | 0.796 s | 0.624 s | 2.91x | 1.02x | 1.28x |
| 1M | 1952 | 2.479 s | 1.149 s | 1.296 s | 0.786 s | 3.15x | 1.46x | 1.65x |

## Low-q, qmax = 2.2 A^-1

| atoms | n_phi | dense total | R-dependent total | sparse source total | combined total | combined vs dense | combined vs R-dependent | combined vs sparse source |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 344 | 0.169 s | 0.075 s | 0.126 s | 0.061 s | 2.78x | 1.23x | 2.08x |
| 250k | 456 | 0.281 s | 0.144 s | 0.225 s | 0.130 s | 2.16x | 1.11x | 1.73x |
| 500k | 564 | 0.396 s | 0.260 s | 0.377 s | 0.136 s | 2.90x | 1.91x | 2.77x |
| 1M | 704 | 0.769 s | 0.643 s | 0.516 s | 0.383 s | 2.01x | 1.68x | 1.35x |

## Peak RSS Delta

Peak RSS delta is the maximum sampled working-set increase during the first
solve call. These runs include the dense amplitude baseline in memory, but each
optional method was measured in its own process to avoid cross-method order
bias.

High-q:

| atoms | n_phi | dense | R-dependent | sparse source | combined |
|---:|---:|---:|---:|---:|---:|
| 100k | 924 | 71.3 MiB | 54.0 MiB | 72.9 MiB | 79.7 MiB |
| 250k | 1242 | 160.7 MiB | 109.2 MiB | 106.9 MiB | 129.0 MiB |
| 500k | 1556 | 304.6 MiB | 172.4 MiB | 189.0 MiB | 179.1 MiB |
| 1M | 1952 | 584.9 MiB | 273.5 MiB | 263.5 MiB | 252.0 MiB |

Low-q:

| atoms | n_phi | dense | R-dependent | sparse source | combined |
|---:|---:|---:|---:|---:|---:|
| 100k | 344 | 27.1 MiB | 19.1 MiB | 23.3 MiB | 30.2 MiB |
| 250k | 456 | 59.3 MiB | 27.0 MiB | 45.0 MiB | 46.0 MiB |
| 500k | 564 | 110.2 MiB | 54.8 MiB | 79.6 MiB | 69.1 MiB |
| 1M | 704 | 210.8 MiB | 96.0 MiB | 106.9 MiB | 92.8 MiB |

## Analytic Kernel Check

At high-q, the analytic compact kernel is the source of the earlier large
NUFFT-relative speedup. Adding the same kernel option to the combined path gives
the same numerical cutoff result, but the standalone first call includes sparse
profile-cache construction.

High-q timing with `--r-dependent-analytic-kernel`. In this all-method run,
the sparse-source projection benchmark executes before the combined benchmark,
so the combined rows reflect a warm sparse profile cache:

| atoms | n_phi | dense total | R-dependent analytic | sparse source | combined analytic |
|---:|---:|---:|---:|---:|---:|
| 100k | 924 | 0.299 s | 0.106 s | 0.241 s | 0.123 s |
| 250k | 1242 | 0.527 s | 0.142 s | 0.382 s | 0.154 s |
| 500k | 1556 | 1.814 s | 0.425 s | 1.043 s | 0.387 s |
| 1M | 1952 | 2.411 s | 0.717 s | 1.306 s | 0.608 s |

The 1M NUFFT comparison run used `--nufft-q-block-size 2`:

| method | first total | cached total | speedup vs chunked NUFFT |
|---|---:|---:|---:|
| R-dependent analytic | 0.550 s | 0.494 s | 62.2x first |
| combined analytic | 1.037 s | 0.514 s | 33.0x first, 66.6x cached |

The standalone combined first call is slower because it builds the sparse
`(element, R)` profile index. After that cache is populated, the solve time is
competitive with the analytic R-dependent path while using less peak RSS in the
1M high-q isolated memory run:

| method | first total | cached total | peak RSS delta |
|---|---:|---:|---:|
| R-dependent analytic | 0.547 s | 0.410 s | 254.7 MiB |
| combined analytic | 1.095 s | 0.494 s | 193.6 MiB |

## R-Dependent Analytic Follow-Up

The R-dependent analytic cake path now has three extra memory-oriented options:

- reuse an already cached full circular FFT directly as the positive-mode
  source for half-spectrum contraction, avoiding a second compact positive-mode
  copy;
- `r_block_size`, which recomputes rFFT blocks over selected R slabs instead of
  using one cached full/half FFT source;
- `fused_analytic_kernel`, which evaluates the Miller analytic kernel inside the
  C++ contraction and avoids materializing the compact `Khat(q,R,h)` array.

1M high-q benchmark conditions:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --skip-nufft --measure-memory --benchmark-r-dependent-cake --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --r-dependent-analytic-kernel
```

| R-dependent analytic variant | first total | cached total | method first | method cached | peak RSS delta |
|---|---:|---:|---:|---:|---:|
| no-copy positive modes | 0.472 s | 0.445 s | 0.350 s | 0.323 s | 40.5 MiB |
| R-block 16 | 1.024 s | 1.002 s | 0.921 s | 0.900 s | 52.8 MiB |
| fused Miller | 0.522 s | 0.455 s | 0.385 s | 0.319 s | 0.9 MiB |
| fused Miller + R-block 16 | 0.967 s | 0.903 s | 0.856 s | 0.792 s | 52.0 MiB |

The current CPU default should be the no-copy positive-mode path. It keeps the
fastest first solve in this 1M high-q case and reduces the old R-dependent
analytic method peak from `254.7 MiB` to `40.5 MiB`.

`fused_analytic_kernel=True` is the best memory-constrained option. It reduces
the measured method peak to `0.9 MiB` because the compact analytic `Khat` array
is generated inside contraction rather than stored. The first solve is slightly
slower than no-copy, but the cached solve is essentially tied.

`r_block_size=16` is not a good CPU default in this dense-baseline benchmark:
the full circular FFT has already been cached, so R-block streaming gives up
that cache and recomputes rFFT slabs. It remains useful as a GPU or strict
working-set design point where the full FFT source should not be resident.

With chunked NUFFT included for the fastest no-copy variant:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --benchmark-r-dependent-cake --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --r-dependent-analytic-kernel --nufft-q-block-size 2
```

| method | total | cached total | chunked NUFFT | speedup vs NUFFT |
|---|---:|---:|---:|---:|
| R-dependent analytic no-copy | 0.516 s | 0.539 s | 30.682 s | 59.4x |

## Interpretation

- The combined path is numerically consistent with the R-dependent cake path:
  the amplitude difference versus R-dependent was around `1e-7` to `1e-6`.
- The memory-optimized combined implementation is now the fastest path in the
  500k and 1M high-q cases, while 250k high-q still favors R-dependent cake.
- With analytic kernels enabled, R-dependent cake remains the best one-shot
  1M high-q CPU path. After the no-copy optimization, combined analytic is no
  longer the lower-memory option versus the default R-dependent analytic path;
  fused Miller is the lower-memory R-dependent analytic variant.
- Peak RSS is not universally lower at small sizes, but at 1M it is lower than
  both R-dependent and sparse source in both high-q and low-q isolated runs.
- The present implementation still computes a full beta FFT after sparse source
  projection, then applies R-dependent cutoffs during contraction. A deeper
  selected-h DFT/FFT hybrid could reduce this further.
- For now, keep R-dependent cake as the conservative default for small to
  medium high-q grids and expose combined sparse source/R-dependent as the best
  large-grid experimental contender.

Source files:

- `benchmark_results/physical_scaling_highq_sparse_source_rdep_memory_optimized_repeats2.json`
- `benchmark_results/physical_scaling_highq_sparse_source_rdep_analytic_repeats2.json`
- `benchmark_results/physical_scaling_1m_q6p3_sparse_source_rdep_analytic_nufft_qb2.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_memory_seq.json`
- `benchmark_results/physical_scaling_1m_q6p3_sparse_source_rdep_analytic_memory_seq.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_nocopy_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_rblock16_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_fused_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_fused_rblock16_memory.json`
- `benchmark_results/physical_scaling_1m_q6p3_rdep_analytic_nocopy_vs_nufft.json`
- `benchmark_results/physical_scaling_lowq_sparse_source_rdep_memory_optimized_repeats2.json`
- `benchmark_results/physical_scaling_highq_rdep_memory_isolated.json`
- `benchmark_results/physical_scaling_highq_sparse_source_memory_isolated.json`
- `benchmark_results/physical_scaling_highq_sparse_source_rdep_memory_isolated.json`
- `benchmark_results/physical_scaling_lowq_rdep_memory_isolated.json`
- `benchmark_results/physical_scaling_lowq_sparse_source_memory_isolated.json`
- `benchmark_results/physical_scaling_lowq_sparse_source_rdep_memory_isolated.json`
