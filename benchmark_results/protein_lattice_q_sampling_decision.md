# Protein-lattice q sampling and prepared-path decision

## Fixed-dq q-range sweep (q-block=2 FINUFFT)

| qmax A^-1 | Nq | Nphi | factorized first-total s | chunked FINUFFT wall s | speedup | cross L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2.13 | 14 | 320 | 0.113 | 5.320 | 47.0x | 6.89e-07 |
| 4.06 | 26 | 512 | 0.255 | 17.726 | 69.5x | 7.22e-07 |
| 6.30 | 40 | 720 | 0.511 | 52.481 | 102.8x | 7.50e-07 |
| 8.06 | 51 | 864 | 0.821 | 121.327 | 147.8x | 7.57e-07 |

## Fixed-range q-resolution sweep

| Nq | dq A^-1 | factorized hot s | FINUFFT hot s | hot speedup | timing source |
|---:|---:|---:|---:|---:|---|
| 32 | 0.04194 | 0.239 | 75.115 | 314.3x | single local first/hot case |
| 128 | 0.01024 | 1.170 | 82.440 | 70.4x | single local first/hot case |
| 512 | 0.00254 | 3.143 | 106.945 | 33.5x | prepared fused 10/30 AB/BA paired medians |

## Exact Nq=512 optimization profile

- legacy / prepared hot: `13.913 / 3.143 s` (`4.43x`)
- legacy-prepared complex L2: `4.811e-14`
- coefficient / synthesis: `3.632 / 0.017 s`
- direct / separable lattice setup: `4.259 / 0.209 s` (`20.4x`)
- lattice complex/intensity L2: `1.611e-10 / 1.282e-10`
- prepared 10/30 paired median / p05: `33.480x / 24.243x`
- Prepared local timing gate PASS; independent-machine replication remains pending.
