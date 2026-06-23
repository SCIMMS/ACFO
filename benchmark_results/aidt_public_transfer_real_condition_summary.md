# Public aIDT real-condition transfer-cache benchmark

This note summarizes the public aIDT Diatom I transfer-function benchmark at
larger detector sizes, including the original `700 x 700` public-data condition.

Note: after adding the Torch GPU path, the CPU inverse FFT was corrected to
apply the final inverse transform over the image-plane axes `(y, x)` for
3D spectra shaped as `(y, x, z)`. The GPU rows and corrected CPU reference
below use that convention.

## Condition

| field | value |
| --- | ---: |
| illumination frames | 24 |
| original detector | `700 x 700` |
| objective/source NA | 0.65 |
| wavelength | `0.515 um` |
| medium index | 1.47 |
| z slices | 35 |
| z range | `-25.5:1.5:25.5 um` |

## Large-Size CPU Results

| size | method | one-shot s | run s | speedup vs streaming run | cache MiB | working MiB | rel-L2 note |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | `streaming` | 56.951 | 56.951 | 1.00x | 0.0 | not measured | reference |
| 512 | `geometry_cache` | 35.640 | 16.937 | 3.36x | 354.0 | 354.0 | `n_re` 1.77e-8, `n_im` 6.28e-6 |
| 512 | `blocked_geometry_cache` | 54.212 | 54.212 | 1.05x | 0.0 | 275.5 | exact vs streaming in this run |
| 700 | `streaming` | 104.502 | 104.502 | 1.00x | 0.0 | not measured | reference |
| 700 | `geometry_cache` | 74.052 | 32.986 | 3.17x | 661.7 | 661.7 | `n_re` 1.74e-8, `n_im` 6.28e-6 |
| 700 | `blocked_geometry_cache` | 102.441 | 102.441 | 1.02x | 0.0 | 471.0 | exact vs streaming in this run |

The table above is retained as the CPU cache tradeoff measurement. The corrected
CPU streaming reference measured during GPU validation is `107.99 s` for
the original `700 x 700` condition.

## Torch GPU Geometry Cache

| size | device | setup s | run median s | speedup vs corrected CPU streaming | setup+run speedup | cache MiB | peak allocated MiB | rel-L2 vs CPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | RTX 2070 SUPER | 0.166 | 0.064 | 219.7x | 61.4x | 88.5 | 374.0 | `n_re` 3.15e-8, `n_im` 8.88e-6 |
| 700 | RTX 2070 SUPER | 0.618 | 0.416 | 259.9x | 104.5x | 661.7 | 2767.0 | `n_re` 2.99e-8, `n_im` 8.15e-6 |
| 700 optimized | RTX 2070 SUPER | 0.794 | 0.097 | 1126.0x | 123.0x | 2076.2 | 3849.4 | `n_re` 2.99e-8, `n_im` 8.15e-6 |

## Readout

- The original public aIDT `700 x 700` detector condition is feasible in the CPU NumPy implementation.
- `geometry_cache` is the practical CPU path for real detector size: it cuts repeated-run time from about `104-108 s` to about `33 s` with about `662 MiB` persistent cache.
- `blocked_geometry_cache` is not a speed path in the current CPU implementation. Its role is memory safety when persistent geometry cache is too large.
- Full `prepared_cache` was intentionally skipped at `512/700`; extrapolating from `256` gives about `6.7 GiB` for `700 x 700`, so it is a GPU/large-RAM option rather than the default real-condition CPU path.
- Torch GPU `geometry_cache` is now the speed path: on the local RTX 2070 SUPER,
  the original `700 x 700` public-data condition runs in `0.416 s` after setup.
- The optimized Torch GPU path uses active support RHS, cached support transfer
  functions, and skips output min/max diagnostics. On the same local RTX 2070
  SUPER, the original `700 x 700` public-data condition runs in `0.097 s`
  after setup, or about `10.26` volumes/s for the computational core.

## Artifacts

- `benchmark_results/aidt_public_transfer_prepared_compare_512.md`
- `benchmark_results/aidt_public_transfer_prepared_compare_full700.md`
- `benchmark_results/aidt_public_transfer_prepared_compare.md`
- `benchmark_results/aidt_public_transfer_torch_gpu_256.md`
- `benchmark_results/aidt_public_transfer_torch_gpu_full700_compare.md`
- `benchmark_results/aidt_public_transfer_torch_gpu_optimized_full700_compare.md`
- `benchmark_results/aidt_torch_gpu_optimization_summary.md`
