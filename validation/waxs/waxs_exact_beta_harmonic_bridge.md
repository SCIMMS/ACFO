# WAXS exact-beta harmonic bridge

Detector azimuth 수는 720으로 고정하고 per-atom beta를 직접 harmonic phase에 넣었다.

- exact-coordinate harmonic vs direct NDFT complex L2: `1.010e-12`
- maximum retained harmonic: `350` / target Nyquist `360`
- exact-coordinate harmonic time: `0.102 s`
- direct NDFT time: `0.943 s`

| R/z bin (nm) | complex L2 | pixel intensity L2 | ring L2 | NCC | time s |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 9.067e-01 | 7.831e-01 | 2.289e-01 | 0.341727 | 0.095 |
| 0.00625 | 6.634e-02 | 6.200e-02 | 8.009e-03 | 0.996483 | 0.088 |
| 0.00078125 | 8.155e-03 | 7.786e-03 | 1.046e-03 | 0.999945 | 0.089 |

- exact-coordinate bridge: **PASS**
- fine R/z with exact beta, pixel <=1%: **PASS**
- 이는 source-coordinate와 detector-Nphi 분리가 수학적으로 가능하다는 proof-of-concept다.
- 현재 Python 경로는 O(Nsource x Nq x Nharmonic) reference이며 production 성능 주장이 아니다.
