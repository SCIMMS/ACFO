# Sparse Source-Projection Full Benchmark

Date: 2026-06-15

This benchmark compares the dense circular solver, the R-dependent cake-map
solver, and the sparse source-projection solver on the same physical grids.
The sparse source-projection path uses the C++ source projection builder with
`profile_chunk_size=64`.

## Commands

High-q solver comparison without NUFFT:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 2 --skip-nufft --benchmark-r-dependent-cake --benchmark-sparse-source-projection --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_highq_sparse_source_rdep_full.json
```

High-q comparison with memory-safe chunked NUFFT:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 6.3 --q-unit inv_angstrom --nq 40 --repeats 1 --nufft-q-block-size 2 --benchmark-r-dependent-cake --benchmark-sparse-source-projection --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_highq_sparse_source_rdep_nufft_qb2_full.json
```

Low-q comparison with NUFFT:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_physical_scaling.py --atoms 100000 250000 500000 1000000 --qmax 2.2 --q-unit inv_angstrom --nq 40 --repeats 2 --benchmark-r-dependent-cake --benchmark-sparse-source-projection --r-dependent-margin 16 --r-dependent-cutoff-bin-size 16 --out benchmark_results\physical_scaling_lowq_sparse_source_rdep_full.json
```

## High-q, qmax = 6.3 A^-1

NUFFT used `--nufft-q-block-size 2`.

| atoms | n_phi | dense total | R-dependent total | sparse source total | NUFFT | R-dependent vs NUFFT | sparse source vs NUFFT | R-dependent intensity error | sparse amplitude error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 924 | 0.2899 s | 0.1657 s | 0.1963 s | 5.3975 s | 32.6x | 27.5x | 1.73e-7 | 1.67e-7 |
| 250k | 1242 | 0.6050 s | 0.2755 s | 0.3172 s | 9.0013 s | 32.7x | 28.4x | 2.66e-6 | 1.05e-6 |
| 500k | 1556 | 1.6634 s | 0.5542 s | 0.7738 s | 16.0031 s | 28.9x | 20.7x | 3.86e-7 | 4.57e-7 |
| 1M | 1952 | 2.2734 s | 1.0801 s | 1.1507 s | 29.1283 s | 27.0x | 25.3x | 4.77e-7 | 2.08e-7 |

## Low-q, qmax = 2.2 A^-1

| atoms | n_phi | dense total | R-dependent total | sparse source total | NUFFT | R-dependent vs NUFFT | sparse source vs NUFFT | R-dependent intensity error | sparse amplitude error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 344 | 0.1049 s | 0.0846 s | 0.1119 s | 0.1789 s | 2.1x | 1.6x | 2.17e-8 | 1.92e-7 |
| 250k | 456 | 0.2360 s | 0.1649 s | 0.2141 s | 0.3701 s | 2.2x | 1.7x | 1.76e-7 | 5.79e-7 |
| 500k | 564 | 0.5015 s | 0.2571 s | 0.3859 s | 0.6176 s | 2.4x | 1.6x | 4.37e-8 | 2.46e-7 |
| 1M | 704 | 0.8186 s | 0.4188 s | 0.6181 s | 1.0768 s | 2.6x | 1.7x | 7.64e-8 | 2.84e-7 |

## Interpretation

- Sparse source projection is a real high-q acceleration over dense circular
  contraction: 1.5x to 2.2x over dense total time in the high-q NUFFT run.
- The R-dependent cake-map solver remains the fastest full-2D path in this
  physical-grid benchmark, especially at 500k and 1M atoms.
- Sparse source projection is still useful because it is exact relative to the
  same dense binned circular solver up to roundoff-level complex64 differences,
  while R-dependent cake-map speed comes with controlled harmonic cutoff error.
- Low-q gains are smaller, as expected; the sparse source path is mainly a
  high-q/fine-grid optimization.

Source files:

- `benchmark_results/physical_scaling_highq_sparse_source_rdep_nufft_qb2_full.json`
- `benchmark_results/physical_scaling_highq_sparse_source_rdep_full.json`
- `benchmark_results/physical_scaling_lowq_sparse_source_rdep_full.json`
- `benchmark_results/physical_scaling_sparse_source_full_summary.csv`
