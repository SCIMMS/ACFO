# Million-atom repeated-crystal 10/30 AB/BA timing (prepared_fused)

- status: `complete`
- atoms / targets: `1,001,000 / 393,216`
- warm-up pairs / measured pairs: `10 / 30`
- factorized median: `3.143 s`
- FINUFFT median: `106.945 s`
- paired speedup median / p05: `33.480x / 24.243x`
- cross complex/intensity L2: `3.156e-07 / 1.146e-07`
- local timing gate: **PASS**

| pair | order | factorized s | FINUFFT s | paired speedup |
|---:|---|---:|---:|---:|
| 1 | AB | 3.209 | 76.227 | 23.753x |
| 2 | BA | 3.055 | 111.014 | 36.333x |
| 3 | AB | 3.134 | 105.712 | 33.729x |
| 4 | BA | 3.307 | 118.804 | 35.929x |
| 5 | AB | 3.151 | 111.919 | 35.517x |
| 6 | BA | 3.019 | 132.189 | 43.788x |
| 7 | AB | 3.201 | 107.577 | 33.607x |
| 8 | BA | 3.695 | 102.891 | 27.847x |
| 9 | AB | 3.380 | 115.248 | 34.101x |
| 10 | BA | 4.050 | 105.853 | 26.134x |
| 11 | AB | 4.504 | 106.314 | 23.607x |
| 12 | BA | 3.161 | 121.118 | 38.313x |
| 13 | AB | 3.127 | 115.655 | 36.987x |
| 14 | BA | 3.747 | 97.283 | 25.961x |
| 15 | AB | 3.418 | 84.905 | 24.842x |
| 16 | BA | 2.997 | 87.792 | 29.295x |
| 17 | AB | 3.121 | 95.360 | 30.559x |
| 18 | BA | 3.015 | 92.046 | 30.532x |
| 19 | AB | 3.108 | 106.023 | 34.118x |
| 20 | BA | 3.597 | 116.675 | 32.438x |
| 21 | AB | 3.647 | 112.863 | 30.949x |
| 22 | BA | 3.232 | 102.010 | 31.567x |
| 23 | AB | 3.122 | 113.803 | 36.446x |
| 24 | BA | 3.028 | 100.068 | 33.050x |
| 25 | AB | 3.130 | 116.422 | 37.194x |
| 26 | BA | 3.115 | 103.884 | 33.353x |
| 27 | AB | 3.101 | 113.873 | 36.719x |
| 28 | BA | 3.002 | 122.330 | 40.746x |
| 29 | AB | 3.170 | 95.614 | 30.165x |
| 30 | BA | 3.041 | 108.917 | 35.814x |

- Factorized backend: `prepared_fused`; lattice backend: `separable`.
- The factorized hot path recomputes the unit-cell amplitude while reusing its explicit prepared state and finite lattice factor when selected.
- FINUFFT reuses four element-specific type-3 plans at eps=1e-6.
- Direct NDFT correctness is established by the separate q=3/subset control.
- This remains a same-machine result until independently repeated.
