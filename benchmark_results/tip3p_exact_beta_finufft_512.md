# TIP3P exact-beta vs FINUFFT at Nq=512

- atoms / targets: `50,430 / 393,216`
- FINUFFT eps / threads: `1e-06 / 4`

| method | setup s | first/hot s | peak RSS delta MiB |
|---|---:|---:|---:|
| cpp_fused exact-beta | 0 | 38.373 | 34.2 |
| FINUFFT reusable plan | 0.522 | 0.438 / 0.387 | 43.8 setup |

- fused vs FINUFFT complex/intensity L2: `3.956e-07 / 3.923e-07`
- FINUFFT/fused hot-time ratio: `0.010x`
- first-total FINUFFT/fused ratio: `0.025x`
- benchmark gate: **PASS**
- comparative performance: **FAIL**

Direct NDFT remains the small-case correctness oracle. At Nq=512 the two optimized methods are cross-checked, not promoted to an independent oracle.
