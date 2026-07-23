# High-q threshold and censored FINUFFT strategy

- selected q block: `2`
- time threshold: `180 s`
- holdout extrapolation gate: `PASS`

## Equal-width q-window position sweep

| q window (A^-1) | Nq/Nphi | ACFO first-total s | FINUFFT measured s | speedup | peak RSS / total | complex L2 |
|---|---:|---:|---:|---:|---:|---:|
| 0.05-1.35 | 16/240 | 0.051 | 4.041 | 79.2x | 4.1% | 4.16e-07 |
| 1.70-3.00 | 16/432 | 0.070 | 11.277 | 160.7x | 8.2% | 1.70e-06 |
| 3.35-4.65 | 16/576 | 0.096 | 27.186 | 282.6x | 15.4% | 1.69e-06 |
| 5.00-6.30 | 16/720 | 0.133 | 41.809 | 314.2x | 23.7% | 1.57e-06 |
| 6.70-8.00 | 16/864 | 0.151 | 73.462 | 487.8x | 33.1% | 1.71e-06 |

## High-q resolution sweep

| Nq/Nphi | ACFO first-total s | FINUFFT measured s | status | measured speedup/lower bound | extrapolated full s |
|---|---:|---:|---|---:|---:|
| 32/864 | 0.270 | 145.442 | complete | 538.2x | - |
| 64/864 | 0.530 | 181.665 | censored | 343.0x | 292.1 |
| 128/864 | 1.063 | - | ACFO only | - | 584.1 |
| 256/864 | 2.561 | - | ACFO only | - | 1168.3 |
| 512/864 | 3.845 | - | ACFO only | - | 2336.5 |

## Interpretation boundary

- Measured rows and timeout-derived lower bounds are primary evidence.
- Extrapolated rows are secondary and must remain visually and verbally distinct.
- Timeout or memory censoring is not converted into an exact FINUFFT runtime.

Overall gate: **PASS**
