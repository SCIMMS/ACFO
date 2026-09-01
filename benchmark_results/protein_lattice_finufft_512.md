# Perfect protein crystal crossover vs FINUFFT, Nq=512

- unit atoms: `8,008`
- targets: `393,216`
- FINUFFT eps / threads: `1e-06 / 4`

| case | atoms | factorized first/hot s | FINUFFT first/hot s | factorized speedup first/hot | FINUFFT execute peak RSS delta | complex L2 |
|---|---:|---:|---:|---:|---:|---:|
| 3x3x3 | 216,216 | 14.607/14.075 | 7.436/6.377 | 0.509x/0.453x | 1899.4 MiB | 3.236e-07 |
| 5x5x5 | 1,001,000 | 16.331/12.624 | 82.341/87.375 | 5.042x/6.921x | 5912.6 MiB | 3.156e-07 |

- crossover detected: **PASS**
- >=1M structured-regime performance gate: **PASS**
- The 216k case favors FINUFFT; the 1.001M exact repeated crystal favors the factorized path.
- This is a standard crystallographic specialization and a narrow regime claim, not a general ACFO or novelty claim.
- Each hot timing is one measured repeat; independent-machine and repeated AB/BA timing remain pending.
