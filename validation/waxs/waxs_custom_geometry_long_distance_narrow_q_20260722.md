# Protein nanocrystal ACFO vs FINUFFT: protein_nanocrystal_lysozyme_1iee_3x3x3_fixed

- source mode: `same_binned`; sources `215,944`; full/active targets `11,520/9,008`
- detector: `custom long-distance chamber, narrow-q partial arcs` (rectangular_flat_detector); active fraction `0.782`; outer-ring fraction `0.042`
- ACFO first/cached: `23.839/23.839 s`
- FINUFFT 4-thread setup/first/cached: `0.000/11.634/11.634 s`
- warm speedup FINUFFT/ACFO: `0.49x`
- complex/intensity/ring L2: `2.675e-07` / `3.692e-07` / `3.519e-07`
- q-row intensity relative L2 median/p99: `2.483e-07` / `1.477e-06`
- break-even repeat: `None`

| T | ACFO total s | FINUFFT total s | FINUFFT/ACFO |
|---:|---:|---:|---:|
| 1 | 23.891 | 11.634 | 0.49x |
| 10 | 238.442 | 116.335 | 0.49x |
| 100 | 2383.956 | 1163.350 | 0.49x |

T=10/100 totals are projections from measured setup, first, and cached medians; they are not 10/100 fully executed workflows.
GPU baseline: unavailable: cufinufft 2.5.1 DLL dependency load failure.
