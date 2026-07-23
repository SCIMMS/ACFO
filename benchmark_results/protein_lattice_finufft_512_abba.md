# Million-atom repeated-crystal 10/30 AB/BA timing

- status: `complete`
- atoms / targets: `1,001,000 / 393,216`
- warm-up pairs / measured pairs: `10 / 30`
- factorized median: `13.913 s`
- FINUFFT median: `98.985 s`
- paired speedup median / p05: `7.138x / 5.817x`
- cross complex/intensity L2: `3.156e-07 / 1.146e-07`
- local timing gate: **PASS**

| pair | order | factorized s | FINUFFT s | paired speedup |
|---:|---|---:|---:|---:|
| 1 | AB | 12.339 | 72.571 | 5.881x |
| 2 | BA | 13.709 | 117.805 | 8.593x |
| 3 | AB | 14.172 | 106.781 | 7.535x |
| 4 | BA | 13.143 | 75.752 | 5.764x |
| 5 | AB | 14.901 | 99.204 | 6.657x |
| 6 | BA | 16.044 | 89.313 | 5.567x |
| 7 | AB | 14.394 | 88.333 | 6.137x |
| 8 | BA | 12.658 | 95.143 | 7.517x |
| 9 | AB | 15.639 | 98.621 | 6.306x |
| 10 | BA | 12.537 | 81.576 | 6.507x |
| 11 | AB | 13.879 | 104.425 | 7.524x |
| 12 | BA | 13.991 | 102.330 | 7.314x |
| 13 | AB | 15.279 | 99.122 | 6.487x |
| 14 | BA | 12.420 | 88.062 | 7.090x |
| 15 | AB | 13.050 | 102.326 | 7.841x |
| 16 | BA | 15.087 | 98.847 | 6.552x |
| 17 | AB | 15.112 | 117.362 | 7.766x |
| 18 | BA | 12.965 | 97.604 | 7.529x |
| 19 | AB | 13.645 | 113.088 | 8.288x |
| 20 | BA | 12.761 | 102.858 | 8.060x |
| 21 | AB | 14.281 | 117.414 | 8.222x |
| 22 | BA | 14.859 | 101.148 | 6.807x |
| 23 | AB | 15.073 | 101.443 | 6.730x |
| 24 | BA | 12.609 | 89.095 | 7.066x |
| 25 | AB | 13.724 | 115.149 | 8.390x |
| 26 | BA | 13.035 | 98.638 | 7.567x |
| 27 | AB | 13.968 | 122.619 | 8.778x |
| 28 | BA | 14.192 | 91.734 | 6.464x |
| 29 | AB | 13.946 | 89.850 | 6.442x |
| 30 | BA | 13.282 | 95.454 | 7.187x |

- The factorized hot path recomputes the unit-cell amplitude and reuses only the finite lattice factor.
- FINUFFT reuses four element-specific type-3 plans at eps=1e-6.
- Direct NDFT correctness is established by the separate q=3/subset control.
- This remains a same-machine result until independently repeated.
