# Protein nanocrystal ACFO vs FINUFFT: protein_nanocrystal_lysozyme_1iee_3x3x3_fixed

- source mode: `same_binned`; sources `216,216`; full/active targets `69,120/61,344`
- detector: `custom close chamber, wide-q partial arcs` (rectangular_flat_detector); active fraction `0.887`; outer-ring fraction `0.039`
- ACFO first/cached: `24.366/24.366 s`
- FINUFFT 4-thread setup/first/cached: `0.000/34.199/34.199 s`
- warm speedup FINUFFT/ACFO: `1.40x`
- complex/intensity/ring L2: `3.548e-07` / `1.847e-07` / `5.599e-08`
- q-row intensity relative L2 median/p99: `2.284e-06` / `2.967e-05`
- break-even repeat: `1`

| T | ACFO total s | FINUFFT total s | FINUFFT/ACFO |
|---:|---:|---:|---:|
| 1 | 24.675 | 34.199 | 1.39x |
| 10 | 243.970 | 341.993 | 1.40x |
| 100 | 2436.914 | 3419.934 | 1.40x |

T=10/100 totals are projections from measured setup, first, and cached medians; they are not 10/100 fully executed workflows.
GPU baseline: unavailable: cufinufft 2.5.1 DLL dependency load failure.
